import re
import operator
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import concat, col, lit, udf, lower, explode
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.types import *


conf = SparkConf().setAppName('Entity Resolution')
sc = SparkContext(conf=conf)
sqlCt = SQLContext(sc)


class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):

        concat_df = df.withColumn("concat_col", concat(col(cols[0]), lit(" "), col(cols[1])))

        lower_df = concat_df.withColumn("lower_col", lower(concat_df.concat_col))

        Tokenizer = udf(lambda x: re.split(r'\W+', x.strip()), ArrayType(StringType()))
        token_df = lower_df.withColumn("token_cols", Tokenizer(lower_df.lower_col))

        stopWords = self.stopWordsBC
        stop_words_remover = udf(lambda x: [item for item in x if item not in stopWords], ArrayType(StringType()))
        stop_words_df = token_df.withColumn("joinKey", stop_words_remover(token_df.token_cols))

        return stop_words_df.drop("concat_col").drop("lower_col").drop("token_cols")

    def filtering(self, df1, df2):

        df1_explode = df1.withColumn("token", explode(df1.joinKey))
        df2_explode = df2.withColumn("token", explode(df2.joinKey))

        candDF = (df2_explode.join(df1_explode, df1_explode.token == df2_explode.token).
                  select(df1_explode.id.alias("id1"), df1_explode.joinKey.alias("joinKey1"),
                         df2_explode.id.alias("id2"), df2_explode.joinKey.alias("joinKey2")).distinct())

        return candDF

    def verification(self, candDF, threshold):

        Jaccard_udf = udf(lambda x, y: len(set(x).intersection(set(y))) / float(len(set(x).union(set(y)))), FloatType())
        resultDF = candDF.withColumn("jaccard", Jaccard_udf(candDF.joinKey1, candDF.joinKey2))

        return resultDF.where(resultDF.jaccard >= threshold)

    def evaluate(self, result, groundTruth):

        R = len(result)
        T = len([e for e in result if e in groundTruth])
        A = len(groundTruth)

        precision = 0.0
        recall = 0.0
        fmeasure = 0.0

        # Formula for Precision
        if T > 0:
            precision = T / float(R)

        # Formula for Recal
        if A > 0:
            recall = T / float(A)

        # Formula for FMeasure
        if (precision > 0.0 or recall > 0.0):
            fmeasure = (2 * precision * recall) / (precision + recall)

        return precision, recall, fmeasure

    def jaccardJoin(self, cols1, cols2, threshold):

        new_DF1 = self.preprocessDF(self.df1, cols1)
        new_DF2 = self.preprocessDF(self.df2, cols2)
        print "Before filtering: %d pairs in total" % (self.df1.count() * self.df2.count())

        candDF = self.filtering(new_DF1, new_DF2)
        print "After Filtering: %d pairs left" % (candDF.count())

        resultDF = self.verification(candDF, threshold)
        print "After Verification: %d similar pairs" % (resultDF.count())

        return resultDF

    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    er = EntityResolution("amazon-google-sample/Amazon_sample", "amazon-google-sample/Google_sample",
                          "amazon-google-sample/stopwords.txt")
    #er = EntityResolution("amazon-google/Amazon", "amazon-google/Google", "amazon-google/stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = sqlCt.read.parquet("amazon-google-sample/Amazon_Google_perfectMapping_sample") \
                            .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    #groundTruth = sqlCt.read.parquet("amazon-google/Amazon_Google_perfectMapping") \
    #                      .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print "(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth)