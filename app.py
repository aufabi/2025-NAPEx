import streamlit as st
from google.cloud import storage, aiplatform
from google.oauth2 import service_account
import os
import tempfile
from datetime import datetime
import pandas as pd

# Create credentials object from secrets
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp"])

# === Load environment variables ===
PROJECT_ID = st.secrets["app_config"]["PROJECT_ID"]
REGION = st.secrets["app_config"]["REGION"]
BUCKET_NAME = st.secrets["app_config"]["BUCKET_NAME"]
PIPELINE_TEMPLATE_PATH = st.secrets["app_config"]["PIPELINE_TEMPLATE_PATH"]
SERVICE_ACCOUNT = st.secrets["app_config"]["SERVICE_ACCOUNT"]

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []

# === UI Header ===
st.image("logo.png")
st.markdown("<h2 style='text-align: center; color: #00A39D;'>NAPEx – NCMS AI Pipeline Experiment</h2>", unsafe_allow_html=True)
st.write("Upload customer cashflow `.csv` files to trigger AutoML forecasting pipelines.")

# === File uploader ===
training_budget_hours = st.number_input("Training Budget (hours)", min_value=1.0, max_value=100.0, value=1.0, step=0.5)
forecast_horizon = st.number_input("Forecast Horizon (days)", min_value=1, max_value=365, value=30)
uploaded_files = st.file_uploader("Upload one or more .csv files", type="csv", accept_multiple_files=True)

# === Trigger pipelines ===
if st.button("🚀 Trigger AutoML Pipelines") and uploaded_files:
    with st.spinner("Processing and triggering pipelines..."):
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(BUCKET_NAME)
        aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)

        processed_customers = {res['Customer'] for res in st.session_state.results}

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            customer_name = os.path.splitext(filename)[0]

            if customer_name in processed_customers:
                st.warning(f"Pipeline for `{customer_name}` has already been triggered. Skipping.")
                continue

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
            job_name = f"pipeline-automl-cashflow-forecast-{customer_name}-{datetime.utcnow().strftime('%Y%m%d')}"
            pipeline_job = aiplatform.PipelineJob(
                display_name=job_name,
                template_path=PIPELINE_TEMPLATE_PATH,
                parameter_values={
                    "gcs_source_uri": gcs_uri,
                    "project": PROJECT_ID,
                    "region": REGION,
                    "customer_name": customer_name,
                    "bucket_name": BUCKET_NAME,
                    "training_budget_milli_node_hours": int(training_budget_hours * 1000),
                    "forecast_horizon": forecast_horizon
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

            st.session_state.results.append({
                "Customer": customer_name,
                "Status": "✅ Triggered",
                "Link": console_link
            })

if st.session_state.results:
    st.success("Pipeline Trigger History")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(
        df,
        column_config={
            "Link": st.column_config.LinkColumn("View in Vertex AI Console")
        },
        hide_index=True,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='pipeline_results.csv',
        mime='text/csv',
    )

    if st.button("Clear History"):
        st.session_state.results = []
        st.rerun()

else:
    st.info("Please upload at least one `.csv` file to begin.")
