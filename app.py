import streamlit as st
from google.cloud import storage, aiplatform
from google.oauth2 import service_account
import os
import tempfile

# Create credentials object from secrets
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp"])

# === Load environment variables ===
PROJECT_ID = st.secrets["app_config"]["PROJECT_ID"]
REGION = st.secrets["app_config"]["REGION"]
BUCKET_NAME = st.secrets["app_config"]["BUCKET_NAME"]
PIPELINE_TEMPLATE_PATH = st.secrets["app_config"]["PIPELINE_TEMPLATE_PATH"]
SERVICE_ACCOUNT = st.secrets["app_config"]["SERVICE_ACCOUNT"]

# === UI Header ===
st.image("logo.png")
st.markdown("<h2 style='text-align: center; color: #00A39D;'>NAPEx – NCMS AI Pipeline Experiment</h2>", unsafe_allow_html=True)
st.write("Upload customer cashflow `.csv` files to trigger AutoML forecasting pipelines.")

# === File uploader ===
training_budget_hours = st.number_input("Training Budget (hours)", min_value=1.0, max_value=100.0, value=1.0, step=0.5)
uploaded_files = st.file_uploader("Upload one or more .csv files", type="csv", accept_multiple_files=True)

# === Trigger pipelines ===
if st.button("🚀 Trigger AutoML Pipelines") and uploaded_files:
    with st.spinner("Processing and triggering pipelines..."):
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(BUCKET_NAME)
        aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)

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
            job_name = f"pipeline-automl-cashflow-{customer_name}"
            pipeline_job = aiplatform.PipelineJob(
                display_name=job_name,
                template_path=PIPELINE_TEMPLATE_PATH,
                parameter_values={
                    "gcs_source_uri": gcs_uri,
                    "project": PROJECT_ID,
                    "region": REGION,
                    "customer_name": customer_name,
                    "bucket_name": BUCKET_NAME,
                    "training_budget_milli_node_hours": int(training_budget_hours * 1000)
                },
                enable_caching=False
            )
            pipeline_job.run(
                service_account = SERVICE_ACCOUNT,
                sync=False
            )

            # Show success and monitoring link
            job_id = pipeline_job.job_id
            console_link = f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/pipelines/runs/{job_id}?project={PROJECT_ID}"

            st.success(f"✅ Triggered pipeline for `{customer_name}`")
            st.markdown(f"[🔗 View Pipeline for {customer_name} in Vertex AI Console]({console_link})", unsafe_allow_html=True)

else:
    st.info("Please upload at least one `.csv` file to begin.")
