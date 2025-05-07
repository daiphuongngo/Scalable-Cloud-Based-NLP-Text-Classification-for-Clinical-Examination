from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType, StringType
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud import storage
import pickle
import pandas as pd
import numpy as np
import os

# ----------------------------
# SETUP
# ----------------------------
spark = SparkSession.builder.appName("SampleClinicalPrediction").getOrCreate()
nltk.download("punkt")

# GCS paths
GCS_INPUT = "gs://cscie192-phuong-bucket-useast1/clinical-NLP-classification/sample_clinical_input.csv"
GCS_MODEL_PATH = "clinical-NLP-classification/models/model.pkl"
BUCKET_NAME = "cscie192-phuong-bucket-useast1"
LOCAL_MODEL_PATH = "/tmp/model.pkl"

# ----------------------------
# LOAD CSV
# ----------------------------
df = spark.read.option("header", True).csv(GCS_INPUT)
df = df.select("transcription")  # or the appropriate column name

# ----------------------------
# TEXT CLEANING
# ----------------------------
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

clean_udf = udf(clean_text, StringType())
df_cleaned = df.withColumn("clean_text", clean_udf(col("transcription")))

# ----------------------------
# COLLECT CLEAN TEXTS
# ----------------------------
texts = df_cleaned.select("clean_text").rdd.map(lambda row: row["clean_text"]).collect()

# ----------------------------
# LOAD MODEL & TFIDF FROM GCS
# ----------------------------
# Download model.pkl from GCS to local
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
model_blob = bucket.blob(GCS_MODEL_PATH)
model_blob.download_to_filename(LOCAL_MODEL_PATH)

# Load the pickled model and vectorizer
with open(LOCAL_MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)
    clf = model_bundle["model"]
    vectorizer = model_bundle["vectorizer"]
    label_encoder = model_bundle["label_encoder"]

# ----------------------------
# VECTORIZE & PREDICT
# ----------------------------
X = vectorizer.transform(texts)
y_pred = clf.predict(X)
y_labels = label_encoder.inverse_transform(y_pred)

# ----------------------------
# CREATE OUTPUT DATAFRAME
# ----------------------------
results = pd.DataFrame({
    "transcription": texts,
    "predicted_label": y_labels
})

# Show or optionally save to GCS
print(results.head())

# Save locally
results_path = "/tmp/predictions.csv"
results.to_csv(results_path, index=False)

# Upload back to GCS
output_blob = bucket.blob("clinical-NLP-classification/predictions/sample_predictions.csv")
output_blob.upload_from_filename(results_path)
print("Predictions saved to GCS.")
