import pyspark
import seaborn as sns
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col

spark = SparkSession.builder.getOrCreate()
churn = spark.read.csv('cell2celltrain.csv', header = True)

cols_to_drop = ['ThreewayCalls', 'CurrentEquipementDays', 'HandsetRefurbished', 'TruckOwner', 'RVOwner',
                'BuysViaMailOrder', 'OwnsMotorcycle']

# (1) Columns to drop
def drop_columns(spark_df, columns_to_drop):
    
    '''
    desc:
        Drop specified columns from given churn dataframe
    inpt:
        spark_df [df]: PySpark churn dataframe
        columns_to_drop [list]: List of columns to drop from given dataframe
    oupt:
        spark_df [df]: Updated PySpark dataframe not containing given columns
    '''
    
    spark_df = spark_df.drop(*columns_to_drop)
    return spark_df

 # (2) Deal with missing values
churn.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in churn.columns]).show()
churn = churn.filter(churn.ServiceArea.isNotNull())