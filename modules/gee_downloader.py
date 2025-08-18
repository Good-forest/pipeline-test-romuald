import geemap
import numpy as np
import rasterio
import requests
from skimage.transform import resize
from ee import Image, ImageCollection, Filter, Reducer, EEException, Initialize, Authenticate
from pathlib import Path
import geopandas as gpd
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = 'sentinel-yoan'
MAX_CLOUD_COVER = 60  # Seuil de nuages fixé à 60%

# --- Configuration ---
# BANDS = [
#     'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
#     'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT'
# ]  # 13 bandes pour SEN2SR
#
BANDS=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# LENGTH and WIDTH EQUAL | MORE than 128
TARGET_SIZE = 128  # Taille fixe requise par SEN2SR

def init_ee():
    try:
        Initialize(project='sentinel-yoan')
    except EEException:

        Authenticate()
        Initialize(project='sentinel-yoan')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def filter_image(image, total_pixels: int, threshold: float, bands=None) -> bool:
    """Pixel with too much black pixels are not considered."""
    bands = bands[0] if bands is not None else "B2"
    not_black_pixels = (
        image.select(bands)
        .reduceRegion(
            bestEffort=True,
            reducer=Reducer.count(),
            geometry=image.geometry(),
            scale=30)
        .values()
        .get(0)
        .getInfo()
    )
    return (1 - not_black_pixels / total_pixels) <= threshold

def get_pixels_count(image, band="B2"):
    geometry = image.geometry()

    if geometry is None:
        geometry = image.geometry().bounds()
        if geometry is None:
            raise ValueError("Cannot determine a valid geometry for the image")

    band = band
    total_pixels = (
        image.select(band).reduceRegion(
            reducer=Reducer.count(),
            geometry=geometry,
            scale=30,
            bestEffort=True,
            maxPixels=1e13
        ).values().get(0).getInfo()
    )
    return total_pixels

def vegetation_bare_soil_mask(image: Image) -> Image:
    """Create a vegetation mask from the Sentinel-2 image."""
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5))
    return image.updateMask(mask)

def treat_one(img, root, geometry):
    date = img.date().format('YYYY-MM-dd').getInfo()
    cloud_pct = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

    # logger.info(f"Traitement de {date} ({cloud_pct}% nuages)")
    image_cleaned = vegetation_bare_soil_mask(img)
    total_pixels = get_pixels_count(image_cleaned)
    if not filter_image(image_cleaned, total_pixels, threshold=0.6):
        logger.info(f"Image filtrée: {date} ({cloud_pct}% nuages)")
        return
    # logger.info(f"Traitement de {date} ({cloud_pct}% nuages) done")

    url = image_cleaned.getDownloadURL({
        'bands': BANDS,
        'region': geometry,
        'scale': 10,
        'format': 'GEO_TIFF'
    })

    filename = f"{date}_{cloud_pct:.0f}p.tif"
    path = root / filename

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Erreur de téléchargement de {date} ({cloud_pct}% nuages)")

    logger.info(f"{path} ({cloud_pct}% nuages)")
    path.write_bytes(response.content)
    resize_image(path)
    logger.info(f"✓ {filename} ({cloud_pct}% nuages)")
    return path

# use multithreading
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing
from tqdm import tqdm

def download_images(zone_name, roi, start_date, end_date):
    """Télécharge les images pour une période spécifique"""
    selected_tiles = ['31TDL']
    bb = BANDS.copy()
    bb.append('SCL')
    collection = (ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .select(bb)
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(Filter.inList('MGRS_TILE', selected_tiles))
                  .filter(Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER))
                  .sort("system:time_start", True)
                  )

    count = collection.size().getInfo()
    image_list = collection.sort('CLOUDY_PIXEL_PERCENTAGE').toList(count)

    logger.info(f"Trouvé {count} images pour {start_date} à {end_date} (max {MAX_CLOUD_COVER}% nuages)")

    root = Path('data/raw') / zone_name
    root.mkdir(parents=True, exist_ok=True)

    downloaded = []
    workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(treat_one, Image(image_list.get(i)), root, roi.geometry())
            for i in range(count)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            downloaded.append(future.result())
    # downloaded = [treat_one(Image(image_list.get(i)).clip(roi), root, roi.geometry()) for i in range(count)]


    return downloaded

def resize_data(data, size=TARGET_SIZE):
    resized_data = np.zeros((data.shape[0], size, size), dtype=data.dtype)
    for i in range(data.shape[0]):
        resized_data[i] = resize(data[i], (size, size),
                                 order=1, preserve_range=True, anti_aliasing=True)
    return resized_data

def get_new_meta(meta, size=TARGET_SIZE):
    transform = rasterio.Affine(10, 0, meta['transform'][2],
                                     0, -10, meta['transform'][5])
    new_meta = {
        'height': size,
        'width': size,
        'transform': transform
    }
    return new_meta

# original_path = image_path.parent / f"original_{image_path.name}"
# with rasterio.open(original_path, 'w', **meta) as dst:
#     dst.write(data)
def resize_image(image_path, size=TARGET_SIZE):
    with rasterio.open(image_path) as src:
        data = src.read()
        meta = src.meta

    size = max(meta['width'], meta['height'])
    logger.info(f"Redimensionnement de {image_path} ({meta['width']}x{meta['height']} -> {size}x{size})")

    meta.update(get_new_meta(meta, size=size))

    resized_data = resize_data(data, size=size)
    with rasterio.open(image_path, 'w', **meta) as dst:
        dst.write(resized_data)

    logger.info(f"Image redimensionnée à {size}x{size}")

def get_aoi_list(shapefiles_to_process, buffer):
    gdf = gpd.read_file(shapefiles_to_process)
    gdf['geometry'] = gdf['geometry'].buffer(0)

    if not buffer: return geemap.geopandas_to_ee(gdf)

    gdf = gdf.dissolve().to_crs(epsg=3857)
    gdf['geometry'] = gdf['geometry'].buffer(buffer).simplify(tolerance=buffer)
    gdf = gdf.to_crs(epsg=4326)
    return geemap.geopandas_to_ee(gdf)


