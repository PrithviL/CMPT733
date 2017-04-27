from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SQLContext
# from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.functions import udf
import operator

conf = SparkConf().setAppName('Anomaly Detection')
sc = SparkContext(conf=conf)
sqlCt = SQLContext(sc)


class AnomalyDetection():
    def readData(self, filename):
        self.rawDF = sqlCt.read.parquet(filename).cache()

    def cat2Num(self, df, indices):

        first_feature = df.select(df.rawFeatures[indices[0]]).distinct().collect()
        second_feature = df.select(df.rawFeatures[indices[1]]).distinct().collect()

#Creating 2 lists to extract features
        first_feature_lst = []
        second_feature_lst = []

        for row in first_feature:
            first_feature_lst.append(row[0])

        for row in second_feature:
            second_feature_lst.append(row[0])

#using the one hot encoder feature to convert to vectors
        def OneHotEncoder(rawFeatures):
            features = []
            first_col = [0.0] * len(first_feature_lst)
            first_col[first_feature_lst.index(rawFeatures[indices[0]])] = 1.0
            features += first_col
            second_col = [0.0] * len(second_feature_lst)
            second_col[second_feature_lst.index(rawFeatures[indices[1]])] = 1.0
            features += second_col

            for i in range(2, len(rawFeatures)):
                features += [float(rawFeatures[i])]

            return features

        encoded = udf(OneHotEncoder, ArrayType(FloatType(), containsNull=False))

        return df.withColumn('features', encoded(df.rawFeatures))

    def addScore(self, df):
        cluster_count = dict(df.groupBy('prediction').count().collect())

        print ("Cluster count = " + str(cluster_count))

        n_max = max(cluster_count.values())
        n_min = min(cluster_count.values())
        cluster_count_bc = sc.broadcast(cluster_count)
        
        # Calculating the score
        score = udf(lambda x: (1.0 * (n_max - cluster_count_bc.value[x]) / (n_max - n_min)), FloatType())

        return df.withColumn('score', score(df.prediction))

    def detect(self, k, t):
        # Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        # Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        # Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        # Adding the score column to df2; The higher the score, the more likely it is an anomaly
        df3 = self.addScore(df2).cache()
        df3.show()

        return df3.where(df3.score > t)


if __name__ == "__main__":
    ad = AnomalyDetection()
    ad.readData('logs-features')
    anomalies = ad.detect(8, 0.97)
    print ("Anamolies = " + str(anomalies.count()))
    anomalies.show()
