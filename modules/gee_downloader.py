import ee
import requests
import yaml
from pathlib import Path
import os
from datetime import datetime
import logging
import numpy as np
import rasterio  # Import manquant ajouté
from skimage.transform import resize

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ID = 'tidy-bindery-461215-i7'
MAX_CLOUD_COVER = 60  # Seuil de nuages fixé à 60%
BANDS = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
    'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT'
]  # 13 bandes pour SEN2SR
TARGET_SIZE = 128  # Taille fixe requise par SEN2SR

# --- Initialisation Earth Engine ---
def init_ee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except ee.EEException:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize(project=PROJECT_ID)

# --- Fonctions utilitaires ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_images(zone_name, roi, start_date, end_date):
    """Télécharge les images pour une période spécifique"""
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)))
    
    image_list = collection.sort('CLOUDY_PIXEL_PERCENTAGE').toList(100)  # Limite à 100 images
    count = image_list.size().getInfo()
    downloaded = []
    
    logger.info(f"Trouvé {count} images pour {start_date} à {end_date} (max {MAX_CLOUD_COVER}% nuages)")
    
    for i in range(count):
        img = ee.Image(image_list.get(i))
        date = img.date().format('YYYY-MM-dd').getInfo()
        cloud_pct = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        
        try:
            # Téléchargement avec les 13 bandes SANS dimensions
            url = img.getDownloadURL({
                'bands': BANDS,
                'region': roi,
                'scale': 10,
                'format': 'GEO_TIFF'
            })
            
            filename = f"{zone_name}_{date}_{cloud_pct:.0f}p.tif"
            path = Path('data/raw') / filename
            
            response = requests.get(url)
            if response.status_code == 200:
                path.write_bytes(response.content)
                downloaded.append(path)
                logger.info(f"✓ {filename} ({cloud_pct}% nuages)")
                
                # Redimensionnement local à 128x128
                resize_image(path)
                
            else:
                logger.error(f"✗ Erreur {response.status_code} sur {filename}")
                
        except Exception as e:
            logger.error(f"⚠️ Erreur sur {date}: {str(e)}", exc_info=True)
    
    return downloaded

def resize_image(image_path):
    """Redimensionne l'image à 128x128 pixels"""
    with rasterio.open(image_path) as src:
        data = src.read()
        meta = src.meta
        
    # Redimensionnement
    resized_data = np.zeros((data.shape[0], TARGET_SIZE, TARGET_SIZE), dtype=data.dtype)
    for i in range(data.shape[0]):
        resized_data[i] = resize(data[i], (TARGET_SIZE, TARGET_SIZE),
                                 order=1, preserve_range=True, anti_aliasing=True)
    
    # Mise à jour des métadonnées
    meta.update({
        'height': TARGET_SIZE,
        'width': TARGET_SIZE,
        'transform': rasterio.Affine(10, 0, meta['transform'][2], 
                                     0, -10, meta['transform'][5])
    })
    
    with rasterio.open(image_path, 'w', **meta) as dst:
        dst.write(resized_data)
    
    logger.info(f"Image redimensionnée à {TARGET_SIZE}x{TARGET_SIZE}")

# Point d'entrée principal
def main():
    # === MODIFIE ICI TES DATES ET LA ZONE ===
    start_date = '2025-06-01'
    end_date = '2025-06-15'
    zone_name = 'combre_valtin'
    config_path = 'config/zones.yaml'
    # ========================================

    # Initialisation
    init_ee()
    
    # Chargement de la configuration
    config = load_config(config_path)
    
    if zone_name not in config['zones']:
        logger.error(f"Zone '{zone_name}' non trouvée dans le fichier de configuration")
        return
    
    zone = config['zones'][zone_name]
    roi = ee.Geometry.BBox(*zone['roi'])
    
    # Création du dossier de sortie
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Téléchargement
    logger.info(f"Début du téléchargement pour {zone_name} ({start_date} à {end_date})")
    files = download_images(zone_name, roi, start_date, end_date)
    
    logger.info(f"Téléchargement terminé: {len(files)} images sauvegardées")

if __name__ == '__main__':
    main()
