import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.clustering import KMeans
from pyspark.mllib.linalg import Vectors
from functools import partial


def word_to_cluster(row, vocab_dict):
    cluster_vector = []
    review = row[0]
    label = row[1]

    for word in review:
        cluster_Id = vocab_dict.value[word]
        cluster_vector.append(cluster_Id)

    return (cluster_vector, label)


def cluster_frequency_vector(row, cluster_count):
    cluster_freq_vector = [0] * cluster_count
    review_clusterId = row[0]
    label = row[1]

    for clusterId in review_clusterId:
        cluster_freq_vector[clusterId] = cluster_freq_vector[clusterId] + 1

    return (Vectors.dense(cluster_freq_vector), label)


def frequency_vector_DataFrame(trainDF, cluster_count):
    regTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="[^a-z]")
    dfTokenizer = regTokenizer.transform(trainDF)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    df_remover = remover.transform(dfTokenizer)

    # feature extraction using Word2vec
    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="filtered", outputCol="word2vec")
    vectors = word2Vec.fit(df_remover).getVectors()
    vectors_DF = vectors.select(vectors.word, vectors.vector.alias("features"))

    #  DF as kmeans
    kmeans = KMeans().setK(cluster_count).setSeed(1)
    km_model = kmeans.fit(vectors_DF)

    # Broadcast operation after getting the words and predictions
    vocabDF = km_model.transform(vectors_DF).select("word", "prediction")
    vocabDict = dict(vocabDF.rdd.collect())
    vocab_dict = sc.broadcast(vocabDict)

    # Cluster vector is in RDD form
    reviewsDF = df_remover.select(df_remover.filtered, df_remover.label).rdd
    clusterVectorRdd = reviewsDF.map(partial(word_to_cluster, vocab_dict=vocab_dict))


    cluster_frequency_feature_Rdd = clusterVectorRdd.map(partial(cluster_frequency_vector, cluster_count=cluster_count))

    cluster_freqDF = cluster_frequency_feature_Rdd.map(lambda (x, y): Row(x, y)).toDF()
    cluster_freq_featureDF = cluster_freqDF.select(cluster_freqDF._1.alias("features"), cluster_freqDF._2.alias("label"))

    return cluster_freq_featureDF


def main(sc, sqlCt, train_data, test_data):
    cluster_count = 10  #Count of the cluster is set as 10

    # Read training data as a DataFrame
    given_DF = sqlCt.read.json(train_data)
    given_TestDF = sqlCt.read.json(test_data)

    trainDF = given_DF.select(given_DF.reviewText, given_DF.overall.alias("label")).cache()
    testDF = given_TestDF.select(given_TestDF.reviewText, given_TestDF.overall.alias("label")).cache()

    cluster_freq_trainDF = frequency_vector_DataFrame(trainDF, cluster_count)
    cluster_freq_testDF = frequency_vector_DataFrame(testDF, cluster_count)

    # Linear regression
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # pipeline
    pipeline = Pipeline(stages=[lr])

    # define parameters
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(),
                              numFolds=5) # 5 fold cross validation

    cv_model = crossval.fit(cluster_freq_trainDF)

    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")

    # Training data evaluation
    train_prediction = cv_model.transform(cluster_freq_trainDF)
    train_rmse = evaluator.evaluate(train_prediction)

    # Testing data evaluation
    test_prediction = cv_model.transform(cluster_freq_testDF)
    test_rmse = evaluator.evaluate(test_prediction)

    print 'Training Root Mean Square Error: ' + str(train_rmse)
    print 'Testing Root Mean Square Error: ' + str(test_rmse)


if __name__ == "__main__":
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    conf = SparkConf().setAppName('Sentiment Analysis Kmeans clustering')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)
    main(sc, sqlCt, train_data, test_data)
