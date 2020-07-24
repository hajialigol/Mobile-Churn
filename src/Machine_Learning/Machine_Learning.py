# imports
import matplotlib.pyplot as plt
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline


def ml_pipeline(pyspark_df):
    '''
    Desc:
        Takes in a PySpark dataframe and creates a machine learning pipeline
    Inpt:
        pyspark_df [df]: PySpark dataframe to be fit and transformed upon
    Oupt:
        piped_df [df]: PySpark dataframe that has been fitted and transformed
    '''  

    # Create lists for inputs to VectorAssembler() and Pipeline(), respectively
    columns_list = []
    pipeline_stages = []    

    # Loop over columns to get necessary information for VectorAssembler()
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
            
    # Create VectorAssembler object        
    vectorAssembler = VectorAssembler(inputCols = columns_list, outputCol = "features")

    # Create pipeline
    pipeline_stages.append(vectorAssembler)
    pipeline = Pipeline(stages = pipeline_stages)
    piped_df = pipeline.fit(pyspark_df).transform(pyspark_df)

    # Return dataframe that the pipeline has been fitted on
    return piped_df


def roc_plot(fitted_model):
    '''
    Desc:
        Takes in a fitted model and generates the appropriate ROC curve
    Inpt:
        fitted_model [Model]: Fitted model
    '''
    pandas_roc = fitted_model.summary.roc.toPandas()
    plt.plot(pandas_roc['FPR'], pandas_roc['TPR'])
    plt.title("ROC Curve")
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.show()


def test_roc_performance(fitted_model, testing_set, evaluator):
    '''
    Desc:
        Takes in a fitted model and returns the ROC evaluation on the test set
    Inpt:
        fitted_model [Model]: Fitted model
        testing_set [df]: PySpark testing data frame
        evaluator [Evaluate]: Metric evaluation object to evaluate predictions on
    Oupt:
        test_roc [float]: ROC performance on given test set
    '''
    predictions = fitted_model.transform(testing_set)
    test_roc = evaluator.evaluate(predictions)
    return test_roc