import os
import random
import sen2sr
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import mlstac
import logging
import platform

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
RAW_DIR = Path('data/raw') 
PROCESSED_DIR = Path('data/processed')
COMPARISON_DIR = Path('data/comparisons')
MODEL_NAME = "SEN2SRLite/NonReference_RGBN_x4"

MODEL_NAME = "SEN2SRLite"
MODEL_DIR = Path("models") / MODEL_NAME.replace('/', '_')
SAMPLE_SIZE = 10
# REQUIRED_BANDS = [1, 2, 3, 7]  # Indices des bandes: B02(1), B03(2), B04(3), B08(7)
REQUIRED_BANDS = [x for x in range(13)]
REQUIRED_BANDS = [x for x in range(6)]

# BANDS = [
#     'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
#     'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT'
# ]

BANDS=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Création des dossiers
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

def init_model():
    """Initialise le modèle avec gestion robuste des erreurs"""
    logger.info(f"Système: {platform.system()} {platform.machine()}")
    logger.info(f"PyTorch version: {torch.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = mlstac.load("model/SEN2SRLite").compiled_model(device=device)
    # return model, device
    # Vérifier l'environnement

    # Téléchargement si nécessaire
    if not any(MODEL_DIR.iterdir()):
        logger.info(f"Téléchargement du modèle {MODEL_NAME}...")
        mlstac.download(
            file=f"https://huggingface.co/tacofoundation/sen2sr/resolve/main/{MODEL_NAME}/mlm.json",
            output_dir=str(MODEL_DIR),
        )

    logger.info(f"Utilisation du dispositif: {device}")

    model = mlstac.load(str(MODEL_DIR)).compiled_model(device=device)
    model.eval()
    return model, device

def fordead_processing(img_path, enhanced, meta):
    date = img_path.name.split("_")[0]
    folder = PROCESSED_DIR / date
    folder.mkdir(parents=True, exist_ok=True)
    for band in range(enhanced.shape[0]):
        band_name = BANDS[band]
        processed_path =  folder / f"SENTINEL2A_{date}_{band_name}.tif"
        data = enhanced[band]

        # add 1 dimension
        data = data.reshape(1, data.shape[0], data.shape[1])
        with rasterio.open(processed_path, 'w', **meta) as dst:
            dst.write(data)

def process_image(model, device, img_path, fordead_processing=False):
    """Traite une image complète avec gestion d'erreurs"""
    with rasterio.open(img_path) as src:
        raw_data = src.read()
        meta = src.meta

    logger.info(f"Traitement de {img_path.name} (shape: {raw_data.shape})")

    input_data = (raw_data / 10_000).astype(np.float32)
    input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
    input_data = torch.from_numpy(input_data).float().to(device)

    enhanced = sen2sr.predict_large(
        model=model,
        X=input_data,
        overlap=32,
    ).squeeze(0)

    logger.info(f"Shape amélioré: {enhanced.shape}")

    count = 1 if fordead_processing else enhanced.shape[0]
    meta.update({
        'dtype': 'float32',
        'count': count,
        'height': enhanced.shape[1],
        'width': enhanced.shape[2]
    })

    processed_path = PROCESSED_DIR / img_path.name
    with rasterio.open(processed_path, 'w', **meta) as dst:
        dst.write(enhanced)

    comparison_path = COMPARISON_DIR / f"comp_{img_path.stem}.png"
    create_rgb_comparison(input_data, enhanced, comparison_path)

    return processed_path, comparison_path

def normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min + 1e-10)

def sent_band_to_rgb(data):
    return np.stack([
        data[2],  # B04 (Red)
        data[1],  # B03 (Green)
        data[0]   # B02 (Blue)
    ], axis=-1)

def create_rgb_comparison(raw_data, enhanced_data, output_path):
    """Crée une comparaison visuelle RGB"""
    # Composition RGB: [B04, B03, B02]
    raw_rgb = sent_band_to_rgb(raw_data)
    enhanced_rgb = sent_band_to_rgb(enhanced_data)

    _, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(normalize(raw_rgb))
    ax[0].set_title('Original (10m)')
    ax[0].axis('off')

    ax[1].imshow(normalize(enhanced_rgb))
    ax[1].set_title('Enhanced (2.5m)')
    ax[1].axis('off')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

def augment_images(zone_name):
    logger.info("Démarrage du traitement SEN2SR")
    model, device = init_model()
    logger.info("Modèle initialisé avec succès")

    d = (RAW_DIR / zone_name).glob('*.tif')
    all_images = list(d)
    selected_images = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))
    selected_images = all_images
    logger.info(f"{len(selected_images)} images sélectionnées")

    for i, img_path in enumerate(selected_images):
        logger.info(f"\n[{i+1}/{len(selected_images)}] Traitement de {img_path.name}")
        processed_path, comparison_path = process_image(model, device, img_path)
        if processed_path:
            logger.info(f"→ Résultat sauvegardé: {processed_path}")
        if comparison_path:
            logger.info(f"→ Comparaison générée: {comparison_path}")

if __name__ == "__main__":
    augment_images("YOUR_ZONE_NAME")
