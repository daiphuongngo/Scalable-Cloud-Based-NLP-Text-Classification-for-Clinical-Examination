# -*- coding: utf-8 -*-
"""
Created on Wed May  7 02:22:11 2025
@author: phuon
"""

# vertex_train.py

import os
import pickle
import pandas as pd
import glob
import numpy as np
import pyarrow.parquet as pq
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from google.cloud import storage

# GCS bucket and paths
BUCKET_NAME = "cscie192-phuong-bucket-useast1"
GCS_PARQUET_PATH = "clinical-NLP-classification/processed/"
GCS_MODEL_PATH = "clinical-NLP-classification/models/model.pkl"
LOCAL_DATA_DIR = "/tmp/clinical_data/"
LOCAL_MODEL_PATH = "model.pkl"

# Ensure local directory exists
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Download Parquet files from GCS
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blobs = bucket.list_blobs(prefix=GCS_PARQUET_PATH)

print("Downloading Parquet files...")
for blob in blobs:
    if blob.name.endswith(".parquet"):
        local_file = os.path.join(LOCAL_DATA_DIR, os.path.basename(blob.name))
        blob.download_to_filename(local_file)
        print(f"Downloaded {blob.name} to {local_file}")

# Load all Parquet files into pandas
parquet_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.parquet"))
dfs = [pq.read_table(f).to_pandas() for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

# Debug: print sample feature entry
print("Sample rows from 'features' column:")
print(df["features"].head(2).tolist())

# Convert sparse dict features to dense vectors
def sparse_dict_to_dense_vector(sparse_dict):
    dense = np.zeros(sparse_dict["size"], dtype=np.float32)
    dense[sparse_dict["indices"]] = sparse_dict["values"]
    return dense

X = np.array([sparse_dict_to_dense_vector(f) for f in df["features"]])
y = df["label_index"].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=10)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model as .pkl
with open(LOCAL_MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

# Upload model to GCS
model_blob = bucket.blob(GCS_MODEL_PATH)
model_blob.upload_from_filename(LOCAL_MODEL_PATH)

print(f"Model uploaded to GCS at: gs://{BUCKET_NAME}/{GCS_MODEL_PATH}")
