from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark import SparkConf,SparkContext
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier



#test the prediction y_pred
def evaluate_prediction(y_pred, y_test):    
    print('Accuracy Score: %s' % accuracy_score(y_pred, y_test))
    print('Precision Score: %s' % precision_score(y_pred, y_test))
    print('Recall Score: %s' % recall_score(y_pred, y_test))
    print('F1 Score: %s' % f1_score(y_pred, y_test))
    print('Confusion Matrix')
    print(confusion_matrix(y_test,y_pred))
    pass



if __name__ == "__main__":
    # Initializing a Spark session
    spark = SparkSession.builder.master("local").appName("Heart_disease").config("spark.some.config.option","some-value").getOrCreate()

    raw_data = spark.read.format("csv").option("header","true").option("inferSchema", "true").load(r"/temp/input/adultdata_cleaned_2.csv")
    raw_data2 = spark.read.format("csv").option("header","true").option("inferSchema", "true").load(r"/temp/input/adulttest_cleaned_2.csv")

    cols=raw_data.columns
    cols.remove("income50K")

    assembler = VectorAssembler(inputCols=cols,outputCol="features")
    # Now let us use the transform method to transform our dataset
    raw_data=assembler.transform(raw_data)
    raw_data2=assembler.transform(raw_data2)

    standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    raw_data=standardscaler.fit(raw_data).transform(raw_data)
    raw_data2=standardscaler.fit(raw_data2).transform(raw_data2)
    #raw_data.select("features","Scaled_features").show(5)

    train = raw_data
    test = raw_data2
    dataset_size=float(train.select("income50K").count())

    numPositives=train.select("income50K").where('income50K == 1').count()
    per_ones=(float(numPositives)/float(dataset_size))*100
    numNegatives=float(dataset_size-numPositives)
    print('The number of ones are {}'.format(numPositives))
    print('Percentage of ones are {}'.format(per_ones))


    # Feature selection using chisquareSelector
    #css = ChiSqSelector(featuresCol='Scaled_features',outputCol='Aspect',labelCol='income',numTopFeatures = 6)
    #train=css.fit(train).transform(train)
    #test=css.fit(test).transform(test)

    #logistic regression
    #lr = LogisticRegression(labelCol="income50K", featuresCol="Scaled_features",maxIter=20)
    dt = DecisionTreeClassifier(labelCol="income50K", featuresCol="Scaled_features")
    model=dt.fit(train)
    predict_train=model.transform(train)
    predict_test=model.transform(test)
    #predict_test.select("TenYearCHD","prediction").show(10)

    #evaluation
    evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="income50K")
    print('-----------------------------------------------')
    predict_train.select("income50K","rawPrediction","prediction","probability").show(40)
    predict_test.select("income50K","rawPrediction","prediction","probability").show(40)
    print('-----------------------------------------------')
    print("The prediction accuracy for train set is {}".format(evaluator.evaluate(predict_train)))
    print("The prediction accuracy for test set is {}".format(evaluator.evaluate(predict_test)))
    print('-----------------------------------------------')
    
    #evaluation confusion matrix
    label_temp = np.array(predict_test.select('income50K').collect())
    prediction = np.array(predict_test.select('prediction').collect())
    print('For testing dataset')
    print('-----------------------')
    evaluate_prediction(label_temp,prediction)
    print('-----------------------')



    




