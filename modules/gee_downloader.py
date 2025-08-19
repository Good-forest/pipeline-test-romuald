import ee
import requests
import yaml
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import random
import rasterio
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = 'tidy-bindery-461215-i7'
MAX_CLOUD_COVER = 60
BANDS = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
    'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT'
]

def init_ee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except ee.EEException:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize(project=PROJECT_ID)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_roi_from_yaml(zone):
    cadastre_path = Path(zone['cadastre'])
    if cadastre_path.exists():
        gdf = gpd.read_file(cadastre_path)
        if gdf.crs is None:
            gdf.set_crs(epsg=2154, inplace=True)
        gdf = gdf.to_crs(epsg=4326)
        polygons = []
        for geom in gdf.geometry:
            if geom.type == 'Polygon':
                polygons.append(ee.Geometry.Polygon(list(geom.exterior.coords)))
            elif geom.type == 'MultiPolygon':
                for poly in geom.geoms:
                    polygons.append(ee.Geometry.Polygon(list(poly.exterior.coords)))
        if polygons:
            logger.info(f"Utilisation du shapefile {cadastre_path}")
            return polygons
        else:
            logger.warning(f"Aucun polygone trouvé dans {cadastre_path}, fallback sur bbox")
    else:
        logger.warning(f"Shapefile {cadastre_path} introuvable, fallback sur bbox")
    bbox = zone['roi']
    return [ee.Geometry.BBox(*bbox)]

def download_images(zone_name, roi, start_date, end_date):
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)))
    image_list = collection.sort('CLOUDY_PIXEL_PERCENTAGE').toList(100)
    count = image_list.size().getInfo()
    downloaded = []
    logger.info(f"Trouvé {count} images pour {zone_name} ({start_date} à {end_date})")
    for i in range(count):
        img = ee.Image(image_list.get(i))
        date = img.date().format('YYYY-MM-dd').getInfo()
        cloud_pct = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        try:
            out_dir = Path('data/raw')
            out_dir.mkdir(parents=True, exist_ok=True)
            url = img.getDownloadURL({
                'bands': BANDS,
                'region': roi,
                'scale': 10,
                'format': 'GEO_TIFF'
            })
            filename = f"{zone_name}_{date}_{cloud_pct:.0f}p.tif"
            path = out_dir / filename
            response = requests.get(url)
            if response.status_code == 200:
                path.write_bytes(response.content)
                downloaded.append(path)
                logger.info(f"✓ {filename} ({cloud_pct}% nuages, dossier: {out_dir.name})")
            else:
                logger.error(f"✗ Erreur {response.status_code} sur {filename}")
        except Exception as e:
            logger.error(f"⚠️ Erreur sur {date}: {str(e)}", exc_info=True)
    return downloaded

def convert_random_tifs_to_png(raw_dir, out_dir, nb_samples=5):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_files = list(raw_dir.glob('*.tif'))
    if not tif_files:
        print("Aucun fichier .tif trouvé dans", raw_dir)
        return
    sample_files = random.sample(tif_files, min(nb_samples, len(tif_files)))
    print(f"Conversion de {len(sample_files)} TIFF en PNG dans {out_dir} :")
    for tif_path in sample_files:
        with rasterio.open(tif_path) as src:
            arr = src.read([1, 2, 3])
            arr = np.clip((arr / np.percentile(arr, 99)) * 255, 0, 255).astype(np.uint8)
            rgb = np.transpose(arr, (1, 2, 0))
        png_path = out_dir / (tif_path.stem + '.png')
        plt.imsave(png_path, rgb)
        print("→", png_path)

def main():
    today = datetime.today()
    start_date = (today - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 ans
    end_date = today.strftime('%Y-%m-%d')
    config_path = 'config/zones.yaml'
    zone_key = 'combre_valtin'

    init_ee()
    config = load_config(config_path)
    if zone_key not in config['zones']:
        logger.error(f"Zone '{zone_key}' non trouvée dans le fichier de configuration")
        return
    zone = config['zones'][zone_key]

    Path('data/raw').mkdir(parents=True, exist_ok=True)

    polygons = get_roi_from_yaml(zone)
    logger.info(f"{len(polygons)} polygone(s) utilisé(s) pour la zone {zone_key}")

    for idx, roi in enumerate(polygons):
        logger.info(f"Téléchargement pour polygone {idx+1}/{len(polygons)}")
        download_images(f"{zone_key}_poly{idx+1}", roi, start_date, end_date)

    logger.info("Téléchargement terminé pour tous les polygones")

    # La conversion automatique en PNG a été supprimée ici

if __name__ == '__main__':
    main()
