#item_item.py

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as func
import sys


def main(argv=None):
    if argv is None:
        inputDir = sys.argv[1]

    conf = SparkConf().setAppName('item-item-cf')
    sc = SparkContext(conf=conf)
    sqlCt = SQLContext(sc)

    text_input = sqlCt.read.text(inputDir + "/MovieLens100K_train.txt")
    train = text_input.map(lambda row: row.value.split("\t")) \
        .map(lambda l: (int(l[0]), int(l[1]), float(l[2]))) \
        .toDF(["userID", "movieID", "rating"])
    train.cache()


    text_input = sqlCt.read.text(inputDir + "/MovieLens100K_test.txt")
    test = text_input.map(lambda row: row.value.split("\t")) \
        .map(lambda l: (int(l[0]), int(l[1]), float(l[2]))) \
        .toDF(["userID", "movieID", "rating"])
    test.cache()

    #calculating the average ratings
    average_rating = train.groupBy("userID").agg(func.mean("rating").alias("avg_rating"))
    final_rating= train.join(average_rating, on="userID")
    rating_deviation = final_rating.withColumn("dev_rating", \
                                             final_rating["rating"] - final_rating["avg_rating"]) \
        .withColumnRenamed("movieID", "movieID_dev") \
        .select("userID", "movieID_dev", "dev_rating", "avg_rating")
    rating_deviation.cache()


    #finding the item item siimilarity
    train_text = train.withColumnRenamed("movieID", "movieID2") \
        .withColumnRenamed("rating", "rating2")
    itempairs = train.join(train_text, on=((train.userID == train_text.userID) & \
                                       (train.movieID < train_text.movieID2)))
    groupby_items = itempairs.groupBy("movieID", "movieID2").agg( \
        func.corr("rating", "rating2").alias("correlation"))
    similar_itempairs = groupby_items.filter("correlation is not Null") \
        .select("movieID", "movieID2", "correlation") \
        .withColumnRenamed("movieID", "movieID1")
    similar_itempairs.cache()

    #setting up the regression evalutator and finding filtered itempairs
    list_result = []
    reg_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        filter_itempair = similar_itempairs.filter("correlation >= " + str(threshold))


        # join test data with ratings from the same users, movieID1 < movieID2
        ID_one = func.udf(lambda id1, id2: id1 if id1 < id2 else id2, IntegerType())
        ID_two = func.udf(lambda id1, id2: id1 if id1 >= id2 else id2, IntegerType())
        test_final_rating = test.join(rating_deviation, "userID") \
            .withColumn("movieID1", ID_one("movieID", "movieID_dev")) \
            .withColumn("movieID2", ID_two("movieID", "movieID_dev"))


        #joining items on similar users
        test_rating_derived = test_final_rating.join(filter_itempair, ["movieID1", "movieID2"]) \
            .select("userID", "movieID", "rating", \
                    "correlation", "dev_rating", "avg_rating")



        # make prediction
        test_rating_group = test_rating_derived.groupBy("userID", "movieID", "rating", "avg_rating") \
            .agg((func.sum(test_rating_derived.dev_rating * test_rating_derived.correlation) \
                  / func.sum(test_rating_derived.correlation)).alias("prediction_dev"))
        make_prediction = test_rating_group.withColumn("prediction", \
                                                            test_rating_group.avg_rating + test_rating_group.prediction_dev) \
            .select("userID", "movieID", "rating", "prediction")


        #final RMSE value
        rmse_final = reg_evaluator.evaluate(make_prediction)
        list_result.append((threshold, rmse_final))

    # Print results
    for items in list_result:
        print("****RMSE is*** = %s" % (items[1]))


if __name__ == "__main__":
    main()

