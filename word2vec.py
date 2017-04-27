from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json, sys
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator


def main(train_data, test_data, sc, sqlContext, output):
    text = sqlContext.read.json(train_data)

    train_df = text.select(text.reviewText, text.overall.alias("label"))

    # regextokenizer to split the words
    regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W|[0-9]")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")

    #word2vec for representing as vectors
    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="filtered", outputCol="features")

    lr = LinearRegression(maxIter=20, regParam=0.1, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[regexTokenizer, remover, word2Vec, lr])

    paramGrid = (ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                 .build())

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(),
                              numFolds=5) # 5 fold cross validation

    cv_model = crossval.fit(train_df)

    # Training data evaluation
    train_prediction = cv_model.transform(train_df)
    print train_prediction.show()
    train_evaluator = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")
    train_rmse = train_evaluator.evaluate(train_prediction)

    text_test = sqlContext.read.json(test_data)
    test_df = text_test.select(text_test.reviewText, text_test.overall.alias("label"))

    # Testing data evaluation
    test_prediction = cv_model.transform(test_df)
    print test_prediction.show()
    test_evaluator = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")
    test_rmse = test_evaluator.evaluate(test_prediction)

    print("Training Root mean square error = " + str(train_rmse))
    print("Testing Root mean square error = " + str(test_rmse))

    #output writen to file
    out_file = open(output, 'w')
    out_file.write(str(train_rmse))
    out_file.write(str(test_rmse))
    out_file.close()


if __name__ == "__main__":
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output = sys.argv[3] #output writen to file
    conf = SparkConf().setAppName('Sentiment Analysis Word2Vec')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    main(train_data, test_data, sc, sqlContext, output)