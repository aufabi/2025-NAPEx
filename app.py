import streamlit as st
from google.cloud import storage, aiplatform
import os
import tempfile
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PIPELINE_TEMPLATE_PATH = os.getenv("PIPELINE_TEMPLATE_PATH")

# === UI Header ===
st.image("logo.png", use_container_width=True)
st.markdown("<h2 style='text-align: center; color: #00A39D;'>NAPEx – NCMS AI Pipeline Experiment</h2>", unsafe_allow_html=True)
st.write("Upload customer cashflow `.csv` files to trigger AutoML forecasting pipelines.")

# === File uploader ===
uploaded_files = st.file_uploader("Upload one or more .csv files", type="csv", accept_multiple_files=True)

# === Trigger pipelines ===
if st.button("🚀 Trigger AutoML Pipelines") and uploaded_files:
    with st.spinner("Processing and triggering pipelines..."):
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        aiplatform.init(project=PROJECT_ID, location=REGION)

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            customer_name = os.path.splitext(filename)[0]

            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            temp_file.write(uploaded_file.getbuffer())
            temp_file.close()

            # Upload to GCS
            gcs_blob_path = f"uploads/{filename}"
            blob = bucket.blob(gcs_blob_path)
            blob.upload_from_filename(temp_file.name)
            gcs_uri = f"gs://{BUCKET_NAME}/{gcs_blob_path}"
            os.remove(temp_file.name)

            # Run pipeline
            job_name = f"automl-cashflow-{customer_name}"
            pipeline_job = aiplatform.PipelineJob(
                display_name=job_name,
                template_path=PIPELINE_TEMPLATE_PATH,
                parameter_values={
                    "gcs_source_uri": gcs_uri,
                    "project": PROJECT_ID,
                    "region": REGION,
                    "customer_name": customer_name,
                    "bucket_name": BUCKET_NAME
                },
                enable_caching=False,
            )
            pipeline_job.run(sync=False)

            # Show success and monitoring link
            job_id = pipeline_job.job_id
            console_link = f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/pipelines/runs/{job_id}?project={PROJECT_ID}"

            st.success(f"✅ Triggered pipeline for `{customer_name}`")
            st.markdown(f"[🔗 View Pipeline for {customer_name} in Vertex AI Console]({console_link})", unsafe_allow_html=True)

else:
    st.info("Please upload at least one `.csv` file to begin.")
