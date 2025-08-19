import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_percentile(img, pmin=2, pmax=98):
    vmin = np.percentile(img, pmin)
    vmax = np.percentile(img, pmax)
    if vmax - vmin < 1e-6:
        return np.zeros_like(img)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

def apply_gamma(img, gamma=2.2):
    img = np.clip(img, 0, 1)
    return np.power(img, 1/gamma)

def save_png(img, path):
    plt.imsave(str(path), np.clip(img, 0, 1))

def process_for_enhanced_and_comparison(tif_path, enhanced_dir, comp_dir):
    enhanced_dir = Path(enhanced_dir)
    comp_dir = Path(comp_dir)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(tif_path) as src:
            if src.count < 3:
                print(f"Fichier {tif_path} ignoré (moins de 3 bandes)")
                return
            arr = src.read([1, 2, 3])
            arr = np.transpose(arr, (1, 2, 0))
    except Exception as e:
        print(f"Erreur lors de la lecture de {tif_path}: {e}")
        return

    # PNG amélioré (pour la reconnaissance de cimes)
    arr_advanced = normalize_percentile(arr)
    arr_advanced = apply_gamma(arr_advanced, gamma=2.2)
    enhanced_png_path = enhanced_dir / (Path(tif_path).stem + "_enhanced.png")
    save_png(arr_advanced, enhanced_png_path)

    # PNG de base (pour la planche de comparaison)
    arr_simple = np.clip((arr / np.percentile(arr, 99)), 0, 1)

    # Planche de comparaison
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(arr_simple)
        ax[0].set_title("PNG de base")
        ax[0].axis('off')
        ax[1].imshow(arr_advanced)
        ax[1].set_title("PNG colorimétrie améliorée")
        ax[1].axis('off')
        plt.tight_layout()
        comp_path = comp_dir / (Path(tif_path).stem + "_comparison.png")
        plt.savefig(comp_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Erreur lors de la création de la comparaison pour {tif_path}: {e}")

    print(f"PNG amélioré : {enhanced_png_path.name} | Comparaison : {comp_path.name}")

def batch_enhanced_and_comparison(processed_dir, enhanced_dir, comp_dir):
    processed_dir = Path(processed_dir)
    tif_files = list(processed_dir.glob("*.tif"))
    print(f"{len(tif_files)} fichiers trouvés dans {processed_dir.resolve()}")
    if not tif_files:
        print("Aucun fichier .tif trouvé à traiter.")
        return
    for tif_path in tif_files:
        process_for_enhanced_and_comparison(tif_path, enhanced_dir, comp_dir)

if __name__ == "__main__":
    batch_enhanced_and_comparison(
        "data/processed",
        "data/colormetry_enhanced_png",   # Dossier PNG améliorés pour IA cimes
        "data/colormetry_comparison"      # Dossier comparaisons visuelles
    )
