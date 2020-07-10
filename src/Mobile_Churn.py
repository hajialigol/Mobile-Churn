import pyspark
import seaborn as sns
import pandas as pd
import os
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.functions import mean as _mean, stddev as _stddev
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

spark = SparkSession.builder.getOrCreate()
churn = spark.read.csv('cell2celltrain.csv', header = True)

cols_to_drop = {'CustomerID','ThreewayCalls', 'CurrentEquipementDays', 'HandsetRefurbished', 'TruckOwner', 'RVOwner',
                'Homeownership', 'BuysViaMailOrder', 'NotNewCellphoneUser', 'OwnsMotorcycle'}

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

churn = drop_columns(churn, cols_to_drop)


# (2) Deal with missing values
churn = churn.filter(churn.ServiceArea.isNotNull())
handset_mean = churn.select(_mean("HandsetPrice").alias("mean")).first()[0]
churn = churn.withColumn("HandsetPrice", when(churn["HandsetPrice"] == "Unknown",
                                              handset_mean).otherwise(churn["HandsetPrice"]))


# (3) Convert columns to correct data type
string_columns = {"Churn", "ServiceArea", "ChildrenInHH", "HandsetWebCapable", "RespondsToMailOffers",
                  "OptOutMailings", "NonUSTravel", "OwnsComputer", "HasCreditCard", "NewCellphoneUser",
                  "MadeCallToRetentionTeam", "CreditRating", "PrizmCode", "Occupation", "MaritalStatus"}

def casting(df, string_cols):   
    '''
    desc:
        Convert select columns of type string into columns of type double.  
    inpt:
        df [df]:
        string_cols [set]:
    oupt:
        df [df]:
    '''
    for column in df.columns:
        if column not in string_cols:
            df = df.withColumn(column, df[column].cast("double"))
    return df

churn = casting(churn, string_columns)


# (4) One-hot encode
def ml_pipeline(pyspark_df):
    
    '''
    desc:
        Takes in a PySpark dataframe and creates a machine learning pipeline
    inpt:
        pyspark_df [df]: PySpark dataframe to be fit and transformed upon
    oupt:
        piped_df [df]: PySpark dataframe that has been fitted and transformed
    '''
    
    pipeline_stages = []
    columns_list = []
    
    for column, dtype in churn.dtypes:
        if dtype == "string":
            indexer_output_name = column + "_index"
            encoder_output_name = column + "_fact"
            indexer = StringIndexer(inputCol = column, outputCol = indexer_output_name)
            encoder = OneHotEncoder(inputCol = indexer_output_name, outputCol = encoder_output_name)
            columns_list.append(encoder_output_name)
            pipeline_stages.append(indexer)
            pipeline_stages.append(encoder)
        else:
            columns_list.append(column)
            
    vectorAssembler = VectorAssembler(inputCols = columns_list, outputCol = "features")
    pipeline_stages.append(vectorAssembler)
    pipeline = Pipeline(stages = pipeline_stages)
    piped_df = pipeline.fit(pyspark_df).transform(pyspark_df)
    return piped_df
