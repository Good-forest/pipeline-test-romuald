from pathlib import Path
import logging
import os
from flask import Flask, request, jsonify
from datetime import datetime
from gee_downloader import download_images, init_ee, get_aoi_list
from sen2sr_processor import augment_images
from drive import upload_files, create_folder_rec

from google.oauth2 import service_account

from googleapiclient.discovery import build

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)
DATE_FORMAT="%Y-%m-%d"
def parse_date(date_str):
    return datetime.strptime(date_str, DATE_FORMAT)

SCOPES = ["https://www.googleapis.com/auth/drive",
          "https://www.googleapis.com/auth/gmail.send"]

SERVICE_ACCOUNT_FILE = './credentials.json'
def init_services():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    delegated_credentials = credentials.with_subject('xavier.louchart@goodforest.fr')
    drive_service = build('drive', 'v3', credentials=delegated_credentials)
    return drive_service

DRIVE_SERVICE = init_services()
DRIVE_FOLDER_ID='1ozRSsiK0iAf4fxvGAMyAJnG49iOVgeil'
RAW_DIR = Path('data') 

@app.route("/upload", methods=["POST"])
def upload():
    zone_name = request.form.get('zone_name')
    if not zone_name: return jsonify(success=False), 400
    root_forest_folder_id = create_folder_rec(DRIVE_SERVICE, ['sen2sr', zone_name], DRIVE_FOLDER_ID)

    # for folder in (RAW_DIR / zone_name).iterdir():
    folder = RAW_DIR / zone_name / 'processed'
    forest_folder_id = create_folder_rec(DRIVE_SERVICE, [folder.name], root_forest_folder_id)
    upload_files(DRIVE_SERVICE, folder, forest_folder_id)
    return jsonify(success=True), 200

@app.route("/sen2sr", methods=["POST"])
def sen2sr():
    zone_name = request.form.get('zone_name')
    logger.info(f"Traitement de {zone_name}")
    augment_images(zone_name)
    logger.info(f"Traitement de {zone_name} terminé")
    return jsonify(success=True), 200


@app.route("/", methods=["POST"])
def main():
    zone_name = request.form.get('zone_name')
    start_str = request.form.get('start_date')
    end_str = request.form.get('end_date')
    start_date = parse_date(start_str)
    end_date = parse_date(end_str)
    roi = request.files.get('roi')
    if not roi:
        return jsonify(success=False, error="'roi' file part is required"), 400
    init_ee()
    roi_gdf = get_aoi_list(roi, buffer=0)
    logger.info(f"Traitement de {zone_name} ({start_date} à {end_date})")

    logger.info(f"Début du téléchargement pour {zone_name} ({start_date} à {end_date})")
    files = download_images(zone_name, roi_gdf, start_date, end_date)

    logger.info(f"Téléchargement terminé: {len(files)} images sauvegardées")
    return jsonify(success=True), 200

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    app.logger.info('Service started...')
else:
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
