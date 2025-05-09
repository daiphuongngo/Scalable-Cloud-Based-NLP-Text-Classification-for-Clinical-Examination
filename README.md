# Scalable-Cloud-Based-NLP-Text-Classification-for-Clinical-Examination


![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master, Data Science**

## CSCI E-192 **Modern Data Analytics** (AWS, GCP)

## Professors: **Edward Sumitra**, **Marina Popova**

## Author: **Dai-Phuong Ngo (Liam)**

## Timeline: January 6th - May 16th, 2025

## Course Goals:

- Understand the scope of Modern Big Data Analytics (BDA) and the current technology landscape in BDA.

- Understand the different stages and business objectives in a modern BDA platform.

- Develop a working knowledge of Spark batch and stream analytics with Python.

- Understand Analytic Data Warehouses with Redshift, Druid, and BigQuery.

- Understand tradeoffs between different storage formats - Parquet, Avro, and Arrow.

- Understand the use of Spark and machine learning libraries to Natural Language Processing (NLP) problems.

- Understand the role of data lakes and delta lakes in a BDA platform.

- Understand the importance of Data Catalogues in a BDA platform.

## Project Objective:

Building a real-time Natural Language Processing feedback processing platform using **Python, PySpark, SQL** integrated with **GCP Vertex AI, BigQuery, GCS, Pub/Sub**, supporting Doctor to determine medical specialties for patients.

## Dataset:

Medical Transcriptions

### Context

Medical data is extremely hard to find due to HIPAA privacy regulations. This dataset offers a solution by providing medical transcription samples.

### Content

This dataset contains sample medical transcriptions for various medical specialties.

https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

## System Architecture Design:

![ChatGPT Image May 8, 2025, 04_54_23 AM](https://github.com/user-attachments/assets/b6bfde4b-ca4b-4e58-a822-a1dbd4d46bd5)

## Data Exploration and Analysis

(to be continued)

Here’s a **summary of highlights** for my **GCP-based Clinical NLP Classification Pipeline**, designed to process clinical notes, classify them, and support real-time alerting:

---

## ✅ **GCP Clinical NLP Classification Pipeline – Full Highlights**

### 🔧 1. **Data Ingestion & Storage**

* **Raw input**: Clinical transcription records in `.csv` (e.g., `sample_clinical_input.csv`)
* **Storage**:

  * Raw and processed data stored in **Google Cloud Storage** at:
    `gs://cscie192-phuong-bucket-useast1/clinical-NLP-classification/`

---

### 🧠 2. **NLP Preprocessing (PySpark)**

* **Component**: `pyspark_nlp_preprocess.py`

* **Goal**: Convert raw clinical text into machine-readable numerical vectors for model training

* **Pipeline Steps**:

  * **Text Cleaning** using Spark UDF:

    * Lowercasing
    * Removing non-alphabetic characters
    * Tokenizing text
    * Removing stopwords (can be extended with domain-specific terms)
  * **Vectorization**:

    * `TfidfVectorizer` transforms cleaned text into sparse vectors (up to 7862 features)
  * **Label Encoding**:

    * Medical specialties are converted to numerical class indices via `LabelEncoder`

* **Output**:

  * Transformed dataset stored in GCS as `.parquet` with:

    * `features` column (TF-IDF sparse vectors)
    * `label_index` column (numerical target)

![NLP process data in PySpark](https://github.com/user-attachments/assets/33560b1d-bc2d-490b-9e86-fe03070766e8)

![GCS processed data](https://github.com/user-attachments/assets/d69e9d39-5f48-49d2-ad50-cb81d8453677)

---

### 🤖 3. **Model Training (Vertex AI / Dataproc)**

* **Code**: `vertex_train.py`
* **Model**: `DecisionTreeClassifier` (`max_depth=10`)
* **Pipeline**:

  * Downloads `.parquet` from GCS
  * Converts sparse vectors to dense arrays
  * Splits into train/test sets
  * Trains model and evaluates performance
  * Uploads `model.pkl` (with vectorizer + label encoder) to GCS

![Dataproc model training p1](https://github.com/user-attachments/assets/dae49db7-b98b-448f-9851-b75c25556112)

![Dataproc model training p2](https://github.com/user-attachments/assets/43f83955-7c7b-4fc5-81d4-52cbe599c14a)

![Dataproc model training p3](https://github.com/user-attachments/assets/720a674b-9788-4048-87ff-305feab50d8f)

![Model saved in GCS](https://github.com/user-attachments/assets/27037c87-12c0-49e7-80cf-234a058bd5cb)

---

### 📈 4. **Prediction Pipeline**

* **Code**: `dataproc_batch_predict.py`
* **Flow**:

  * Reads new `.csv` input file (e.g., clinical transcription)
  * Applies **same NLP preprocessing** (cleaning + TF-IDF)
  * Loads trained model from GCS
  * Predicts class labels
  * Saves results to BigQuery

---

### 🔔 5. **Real-time Alerting (Pub/Sub / Cloud Function – Optional)**

* **Future add-on**:

  * Trigger **Pub/Sub** topic when a `.csv` file lands in `predictions/` folder
  * Cloud Function parses it and sends alerts if certain specialties (e.g., `"Oncology"`) are predicted

---

### 🛠️ 6. **Tech Stack Summary**

* **Compute**: Google Dataproc, Vertex AI
* **Data Tools**: PySpark, Pandas, Scikit-learn, TF-IDF, LabelEncoder
* **Cloud Services**: GCS, Cloud Functions (future), Pub/Sub (future)
* **Model Output**: Pickled model + vocabulary, deployable for inference

