from pyspark.sql.functions import col, count

# (1) Columns to drop_columns
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


def find_null_counts(pyspark_df):
    '''
    desc:
        This function finds the columns that contain "NA"s represented as strings along with their length.
    inpt:
        pyspark_df [df]: Dataframe to find string "NA"s from.
    oupt:
        columns_string_nas [dict]: Dictionary of columns that have "NA"s represented as strings with the
                                 key being the column name and the value being the amount of string "NA"s.
    '''
    columns_string_nas = {}
    for column in pyspark_df.columns:
        amount_string_nas = (pyspark_df.select(column).where(col(column) == "NA")).count()
        if amount_string_nas > 0:
            columns_string_nas[column] = amount_string_nas
    return columns_string_nas


def remove_nulls(pyspark_df, set_of_nulls):
    '''
    desc:
        This function removes "NA"s of type string.
    inpt:
        pyspark_df [df]: PySpark dataframe to remove string nulls from.
        set_of_nulls [set]: Set containing the columns that have "NA"s represented as strings.
    oupt:
        pyspark_df [df]: Updated PySpark dataframe without string "NA"s.
    '''
    for column in set_of_nulls:
        pyspark_df = pyspark_df.filter(col(column) != "NA")
    return pyspark_df


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