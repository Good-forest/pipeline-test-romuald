import os
import random
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
MODEL_DIR = Path("models") / MODEL_NAME.replace('/', '_')
SAMPLE_SIZE = 10
REQUIRED_BANDS = [1, 2, 3, 7]  # Indices des bandes: B02(1), B03(2), B04(3), B08(7)

# Création des dossiers
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

def init_model():
    """Initialise le modèle avec gestion robuste des erreurs"""
    try:
        # Vérifier l'environnement
        logger.info(f"Système: {platform.system()} {platform.machine()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Téléchargement si nécessaire
        if not any(MODEL_DIR.iterdir()):
            logger.info(f"Téléchargement du modèle {MODEL_NAME}...")
            mlstac.download(
                file=f"https://huggingface.co/tacofoundation/sen2sr/resolve/main/{MODEL_NAME}/mlm.json",
                output_dir=str(MODEL_DIR),
            )
        
        # Initialisation du modèle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilisation du dispositif: {device}")
        
        model = mlstac.load(str(MODEL_DIR)).compiled_model(device=device)
        model.eval()
        return model, device
    except Exception as e:
        logger.error(f"Erreur d'initialisation du modèle: {str(e)}", exc_info=True)
        raise

def process_image(model, device, img_path):
    """Traite une image complète avec gestion d'erreurs"""
    try:
        # Chargement des bandes spécifiques
        with rasterio.open(img_path) as src:
            # Lire uniquement les bandes requises
            raw_data = src.read([i+1 for i in REQUIRED_BANDS])
            meta = src.meta
        
        logger.info(f"Traitement de {img_path.name} (shape: {raw_data.shape})")
        
        # Normalisation et préparation
        input_data = (raw_data / 10000).astype(np.float32)
        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Conversion en tensor
        input_tensor = torch.from_numpy(input_data).float().to(device).unsqueeze(0)
        
        # Application du modèle
        with torch.no_grad():
            enhanced = model(input_tensor).squeeze(0).cpu().numpy()
        
        logger.info(f"Shape amélioré: {enhanced.shape}")
        
        # Sauvegarde du résultat
        processed_path = PROCESSED_DIR / f"sr_{img_path.name}"
        
        # Mise à jour des métadonnées
        meta.update({
            'dtype': 'float32',
            'count': enhanced.shape[0],
            'height': enhanced.shape[1],
            'width': enhanced.shape[2]
        })
        
        with rasterio.open(processed_path, 'w', **meta) as dst:
            dst.write(enhanced.astype(np.float32))
        
        # Comparaison
        comparison_path = COMPARISON_DIR / f"comp_{img_path.stem}.png"
        create_rgb_comparison(input_data, enhanced, comparison_path)
        
        return processed_path, comparison_path
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {img_path.name}: {str(e)}", exc_info=True)
        return None, None

def create_rgb_comparison(raw_data, enhanced_data, output_path):
    """Crée une comparaison visuelle RGB"""
    try:
        # Composition RGB: [B04, B03, B02]
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
        
        # Normalisation pour l'affichage
        def normalize(img):
            img_min = np.min(img)
            img_max = np.max(img)
            return (img - img_min) / (img_max - img_min + 1e-10)
        
        # Création de la figure
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

def main():
    logger.info("Démarrage du traitement SEN2SR (RGBN_x4)")
    
    try:
        # Initialisation du modèle
        model, device = init_model()
        logger.info("Modèle initialisé avec succès")
        
        # Sélection aléatoire d'images
        all_images = list(RAW_DIR.glob('*.tif'))
        selected_images = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))
        logger.info(f"{len(selected_images)} images sélectionnées")
        
        # Traitement des images
        for i, img_path in enumerate(selected_images):
            logger.info(f"\n[{i+1}/{len(selected_images)}] Traitement de {img_path.name}")
            processed_path, comparison_path = process_image(model, device, img_path)
            if processed_path:
                logger.info(f"→ Résultat sauvegardé: {processed_path}")
            if comparison_path:
                logger.info(f"→ Comparaison générée: {comparison_path}")
                
    except Exception as e:
        logger.exception("Erreur critique dans le main")
    finally:
        logger.info("Traitement terminé")

if __name__ == "__main__":
    main()
