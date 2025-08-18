from pathlib import Path
import logging
import os
from flask import Flask, request, jsonify
from datetime import datetime
from gee_downloader import download_images, init_ee, get_aoi_list
from sen2sr_processor import augment_images

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)
DATE_FORMAT="%Y-%m-%d"
def parse_date(date_str):
    return datetime.strptime(date_str, DATE_FORMAT)

@app.route("/sen2sr", methods=["POST"])
def sen2sr():
    zone_name = request.form.get('zone_name')

    logger.info(f"Traitement de {zone_name}")
    augment_images(zone_name)
    logger.info(f"Traitement de {zone_name} terminé")
    return jsonify({"status": "FullQueue"}), 200


@app.route("/", methods=["POST"])
def main():
    zone_name = request.form.get('zone_name')
    start_str = request.form.get('start_date')
    end_str = request.form.get('end_date')
    start_date = parse_date(start_str)
    end_date = parse_date(end_str)
    roi = request.files['roi']
    init_ee()
    roi_gdf = get_aoi_list(roi, buffer=0)

    logger.info(f"Traitement de {zone_name} ({start_date} à {end_date})")

    Path('data/raw').mkdir(parents=True, exist_ok=True)

    logger.info(f"Début du téléchargement pour {zone_name} ({start_date} à {end_date})")
    files = download_images(zone_name, roi_gdf, start_date, end_date)

    logger.info(f"Téléchargement terminé: {len(files)} images sauvegardées")
    return jsonify({"status": "FullQueue"}), 200

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    app.logger.info('Service started...')
else:
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
