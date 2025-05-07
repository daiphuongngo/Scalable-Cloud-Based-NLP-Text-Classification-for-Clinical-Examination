from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml import Pipeline

# Create Spark session
spark = SparkSession.builder \
    .appName("ClinicalNLPTextPreprocessing") \
    .getOrCreate()

# Read the dataset from GCS
gcs_path = "gs://cscie192-phuong-bucket-useast1/clinical-NLP-classification/mtsamples.csv"
df = spark.read.option("header", True).csv(gcs_path)

# Drop rows with null in required fields
df_cleaned = df.filter(col("description").isNotNull() & col("medical_specialty").isNotNull())

# NLP pipeline components
tokenizer = Tokenizer(inputCol="description", outputCol="words_token")
remover = StopWordsRemover(inputCol="words_token", outputCol="words_clean")
vectorizer = CountVectorizer(inputCol="words_clean", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")

# Handle nulls in labels safely
indexer = StringIndexer(inputCol="medical_specialty", outputCol="label_index", handleInvalid="skip")

# Create full pipeline
pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, idf, indexer])
model = pipeline.fit(df_cleaned)
df_final = model.transform(df_cleaned)

# Write final processed features to GCS
output_path = "gs://cscie192-phuong-bucket-useast1/clinical-NLP-classification/processed"
df_final.select("features", "label_index").write.mode("overwrite").parquet(output_path)

spark.stop()
