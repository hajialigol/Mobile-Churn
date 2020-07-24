# imports
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline


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
    
    for column, dtype in pyspark_df.dtypes:
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