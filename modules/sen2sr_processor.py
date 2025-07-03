import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import mlstac
import logging
import platform

try:
    from sen2sr import predict_large
except ImportError:
    raise ImportError(
        "Le module 'sen2sr' n'est pas installé. "
        "Installe-le avec : pip install sen2sr mlstac git+https://github.com/ESDS-Leipzig/cubo.git"
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "SEN2SRLite_NonReference_RGBN_x4"
MODEL_DIR = Path("models") / MODEL_NAME
REQUIRED_BANDS = [1, 2, 3, 7]  # B02(1), B03(2), B04(3), B08(7)

def pad_to_multiple(img, multiple=128, mode='reflect'):
    bands, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_before_h = pad_h // 2
    pad_after_h = pad_h - pad_before_h
    pad_before_w = pad_w // 2
    pad_after_w = pad_w - pad_before_w
    pad_width = (
        (0, 0),
        (pad_before_h, pad_after_h),
        (pad_before_w, pad_after_w)
    )
    img_padded = np.pad(img, pad_width, mode=mode)
    return img_padded, pad_before_h, pad_after_h, pad_before_w, pad_after_w

def init_model():
    try:
        logger.info(f"Système: {platform.system()} {platform.machine()}")
        logger.info(f"PyTorch version: {torch.__version__}")

        if not any(MODEL_DIR.iterdir()):
            raise FileNotFoundError(
                f"Le dossier modèle {MODEL_DIR} est vide. "
                "Place les fichiers du modèle SEN2SRLite/NonReference_RGBN_x4 dans ce dossier."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilisation du dispositif: {device}")

        model = mlstac.load(str(MODEL_DIR)).compiled_model(device=device)
        model.eval()
        return model, device
    except Exception as e:
        logger.error(f"Erreur d'initialisation du modèle: {str(e)}", exc_info=True)
        raise

def process_image(model, device, img_path, processed_dir, comparison_dir):
    try:
        with rasterio.open(img_path) as src:
            raw_data = src.read([i+1 for i in REQUIRED_BANDS])
            meta = src.meta

        logger.info(f"Traitement de {img_path.name} (shape: {raw_data.shape})")

        input_data = (raw_data / 10000).astype(np.float32)
        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        input_data, pbh, pah, pbw, paw = pad_to_multiple(input_data, 128)
        input_tensor = torch.from_numpy(input_data).float().to(device)

        with torch.no_grad():
            enhanced_full = predict_large(input_tensor, model).cpu().numpy()

        if pbh + pah + pbw + paw > 0:
            enhanced = enhanced_full[
                :,
                pbh * 4 : enhanced_full.shape[1] - pah * 4 if pah > 0 else None,
                pbw * 4 : enhanced_full.shape[2] - paw * 4 if paw > 0 else None
            ]
        else:
            enhanced = enhanced_full

        logger.info(f"Shape amélioré: {enhanced.shape}")

        processed_path = processed_dir / f"sr_{img_path.name}"
        processed_dir.mkdir(parents=True, exist_ok=True)
        meta.update({
            'dtype': 'float32',
            'count': enhanced.shape[0],
            'height': enhanced.shape[1],
            'width': enhanced.shape[2]
        })

        with rasterio.open(processed_path, 'w', **meta) as dst:
            dst.write(enhanced.astype(np.float32))

        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = comparison_dir / f"comp_{img_path.stem}.png"
        create_rgb_comparison(input_data, enhanced, comparison_path)

        return processed_path, comparison_path
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {img_path.name}: {str(e)}", exc_info=True)
        return None, None

def create_rgb_comparison(raw_data, enhanced_data, output_path):
    try:
        raw_rgb = np.stack([
            raw_data[2],  # B04 (Red)
            raw_data[1],  # B03 (Green)
            raw_data[0]   # B02 (Blue)
        ], axis=-1)

        enhanced_rgb = np.stack([
            enhanced_data[2],  # B04 (Red)
            enhanced_data[1],  # B03 (Green)
            enhanced_data[0]   # B02 (Blue)
        ], axis=-1)

        def normalize(img):
            img_min = np.min(img)
            img_max = np.max(img)
            return (img - img_min) / (img_max - img_min + 1e-10)

        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].imshow(normalize(raw_rgb))
        ax[0].set_title('Original (10m)')
        ax[0].axis('off')

        ax[1].imshow(normalize(enhanced_rgb))
        ax[1].set_title('Enhanced (2.5m)')
        ax[1].axis('off')

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Erreur dans create_rgb_comparison: {str(e)}", exc_info=True)
        return False

def process_folder(model, device, raw_dir, processed_dir, comparison_dir):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    comparison_dir = Path(comparison_dir)
    all_images = list(raw_dir.glob('*.tif'))
    if not all_images:
        logger.warning(f"Aucune image .tif trouvée dans {raw_dir}")
        return
    selected_images = all_images  # Traite toutes les images
    logger.info(f"{len(selected_images)} images sélectionnées dans {raw_dir}")

    for i, img_path in enumerate(selected_images):
        logger.info(f"\n[{i+1}/{len(selected_images)}] Traitement de {img_path.name}")
        processed_path, comparison_path = process_image(model, device, img_path, processed_dir, comparison_dir)
        if processed_path:
            logger.info(f"→ Résultat sauvegardé: {processed_path}")
        if comparison_path:
            logger.info(f"→ Comparaison générée: {comparison_path}")

def main():
    logger.info("Démarrage du traitement SEN2SR (RGBN_x4)")

    try:
        model, device = init_model()
        logger.info("Modèle initialisé avec succès")

        # Traite toutes les images de chaque dossier
        process_folder(
            model, device,
            raw_dir='data/raw/moins20',
            processed_dir='data/processed/moins20',
            comparison_dir='data/comparisons/moins20'
        )
        process_folder(
            model, device,
            raw_dir='data/raw/20a60',
            processed_dir='data/processed/20a60',
            comparison_dir='data/comparisons/20a60'
        )

    except Exception as e:
        logger.exception("Erreur critique dans le main")
    finally:
        logger.info("Traitement terminé")

if __name__ == "__main__":
    main()
