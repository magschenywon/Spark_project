#!/bin/bash
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /temp/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /temp/output
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /temp/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/*cleaned* /temp/input/
/usr/local/spark/bin/spark-submit --conf spark.default.parallelism=3 --master=spark://$SPARK_MASTER:7077 ./part4_spark_dt.py
