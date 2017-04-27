#README
#to run this code: spark-submit als.py/movierecommendation out 'name of output.txt file'
#the output txt file contains the list of the RMSE values arranged accordingly

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
import sys

inputDir = sys.argv[1]
outputDir = sys.argv[2]

conf = SparkConf().setAppName('Collaborative filtering')
sc = SparkContext(conf=conf)
sqlCt = SQLContext(sc)

class Remove_nan(RegressionEvaluator):

    def evaluate(self, dataset):

        def nan_filter(row):

            if str(row)=='nan':
                print "nan:", row
                return 0.0
            else:
                return row
        u = udf(nan_filter, FloatType())

        dataset = dataset.withColumn("prediction_new", u(dataset.prediction))
        cols = dataset.columns
        cols.remove("prediction")
        dataset = dataset.select(cols)
        dataset = dataset.withColumnRenamed("prediction_new","prediction")
        return super(Remove_nan,self).evaluate(dataset)

def getRecords(line):
    words = line.split('\t')
    return (int(words[0]), int(words[1]), float(words[2]))

def getMovieList(line):
    words = line.split('|')
    return (int(words[0]), words[1])

class movieRecALS:
    def __init__(self, trainData, testData, movieList):
        self.trainDf = sc.textFile(trainData).map(getRecords).toDF(['UserID', 'MovieID', 'label']).cache()
        self.testDf = sc.textFile(testData).map(getRecords).toDF(['UserID', 'MovieID', 'label']).cache()
        self.movieDf = sc.textFile(movieList).map(getMovieList).toDF(['MovieID', 'MovieName']).cache()

    def trainALS(self, ranks, iterations):
        for rank in ranks:
            als = ALS(rank=rank, maxIter=iterations, regParam=0.1, userCol="UserID", itemCol="MovieID",ratingCol="label")
            paramGrid = ParamGridBuilder().addGrid(als.rank,[rank]).build()
            crossval = CrossValidator(estimator=als,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=Remove_nan(metricName="rmse", labelCol="label",
                                      predictionCol="prediction"),
                                      numFolds=5)
            self.trainDf.show()
            cvModel = crossval.fit(self.trainDf)
            predictions = cvModel.transform(self.testDf)
            rmse = Remove_nan(metricName="rmse", labelCol="label",
                                        predictionCol="prediction").evaluate(predictions)
            print "****RMSE VALUE IS :*****", rmse
            movieFactors = cvModel.bestModel.itemFactors.orderBy('id').cache()
            movieFactors.show(truncate=False)
            convertToVectors = udf(lambda features: Vectors.dense(features), VectorUDT())
            movieFactors = movieFactors.withColumn("features", convertToVectors(movieFactors.features))
            kmeans = KMeans(k=50, seed=1)
            kModel = kmeans.fit(movieFactors)
            kmeansDF = kModel.transform(movieFactors)
            clusters = [1, 2]
            kmeansDF = kmeansDF.join(self.movieDf, kmeansDF.id == self.movieDf.MovieID).drop('MovieID')
            for cluster in clusters:
                movieNamesDf = kmeansDF.where(col("prediction") == cluster).select("MovieName")
                movieNamesDf.rdd.map(lambda row: row[0]).saveAsTextFile(outputDir + \
                                                                        "Rank" + str(rank) + "Cluster" + str(cluster))

        if __name__ == "__main__":
            mr = movieRecALS(inputDir + "/MovieLens100K_train.txt", inputDir + "/MovieLens100K_test.txt",
                             inputDir + "/u.item")
            ranks = [2, 4, 8, 16, 32, 64, 128, 256]
            iterations = 20
            mr.trainALS(ranks, iterations)
