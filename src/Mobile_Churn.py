from Data_Cleaning.Data_Cleaner import *
from Machine_Learning.Machine_Learning import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, mean as _mean, when, row_number
from pyspark.sql import Window
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune

# Create Sparksession
spark = SparkSession.builder.getOrCreate()

# Initialize data directory
data_directory = ''

# Read data
churn = spark.read.csv(data_directory, header = True)

# columns to drop
cols_to_drop = {'CustomerID','ThreewayCalls', 'CurrentEquipementDays', 'HandsetRefurbished', 'TruckOwner', 'RVOwner',
                'Homeownership', 'BuysViaMailOrder', 'NotNewCellphoneUser', 'OwnsMotorcycle'}

# Drop columns
churn = drop_columns(churn, cols_to_drop)

# Deal with missing values 
churn = churn.filter(churn.ServiceArea.isNotNull())


# HandsetPrice
handset_mean = churn.select(_mean("HandsetPrice").alias("mean")).first()[0]
churn = churn.withColumn("HandsetPrice", when(churn["HandsetPrice"] == "Unknown",
                                              handset_mean).otherwise(churn["HandsetPrice"]))

# Get rid of nulls
null_dict = find_null_counts(churn)
null_set = set(null_dict.keys())
churn = remove_nulls(churn, null_set)

# Columns to cast to different type
string_columns = {"Churn", "ServiceArea", "ChildrenInHH", "HandsetWebCapable", "RespondsToMailOffers",
                  "OptOutMailings", "NonUSTravel", "OwnsComputer", "HasCreditCard", "NewCellphoneUser",
                  "MadeCallToRetentionTeam", "CreditRating", "PrizmCode", "Occupation", "MaritalStatus"}

# Cast given columns to type double
churn = casting(churn, string_columns)

# Change output ame
churn = churn.withColumn("label", col("Churn") == "Yes")

# Cast output to type int
churn = churn.withColumn("label", col("label").cast('int'))

# Generate window
w = Window().orderBy(lit('A'))

# Create column to produce row numbers
churn = churn.withColumn("row_num", row_number().over(w))

# Create machine learning pipeline
piped = ml_pipeline(churn)

# Standardize dataset
training, testing = piped.randomSplit([.75, .25])

# Create evaluator
evaluator = evals.BinaryClassificationEvaluator(metricName = "areaUnderROC")

# Create logistic regression model
lr = LogisticRegression()

# Create cross validation object
cross_validation = tune.CrossValidator(estimator = lr, evaluator = evaluator)

# Train model
lr.fit(training)