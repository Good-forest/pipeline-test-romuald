import streamlit as st

st.title('SEN2SR processing')

BACKEND_ENDPOINT = "https://oauth2.googleapis.com/revoke"

UPLOAD_ENDPOINT = "https://oauth2.googleapis.com/revoke"

start = st.date_input("Start")
end = st.date_input("End")
zone_name = st.text_input("Zone name")
shape = st.file_uploader('Upload a geojson file', type=['geojson'])

DATE_FORMAT = '%Y-%m-%d'
import requests

def run_cloud_run(start, end, zone_name, shape):
    url = "http://localhost:8080"
    data = {
        "start_date": start.strftime(DATE_FORMAT),
        "end_date": end.strftime(DATE_FORMAT),
        "zone_name": zone_name,
    }
    response = requests.post(url, data=data, files={"roi": shape})
    return response.json()

def run_sen2sr(zone_name):
    url = "http://localhost:8080/sen2sr"
    data = {
        "zone_name": zone_name,
    }
    response = requests.post(url, data=data)
    return response.json()

def run_upload(zone_name):
    url = "http://localhost:8080/upload"
    data = {
        "zone_name": zone_name,
    }
    response = requests.post(url, data=data)
    return response.json()

if st.button("Submit"):
    run_cloud_run(start, end, zone_name, shape)
    run_sen2sr(zone_name)

if st.button("Submit 2"):
    run_upload(zone_name)
