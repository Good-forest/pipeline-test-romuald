import rasterio
from pathlib import Path

def filter_valid_tifs(raw_dir, required_band_count=13):
    """
    Liste les fichiers .tif dans raw_dir qui ont exactement le nombre de bandes attendu.

    Args:
        raw_dir (str or Path): Dossier contenant les fichiers .tif à vérifier.
        required_band_count (int): Nombre de bandes attendu (par défaut 13 pour Sentinel-2).

    Returns:
        list of Path: Liste des chemins des fichiers .tif valides.
    """
    raw_dir = Path(raw_dir)
    valid_files = []
    for tif_path in raw_dir.glob('*.tif'):
        try:
            with rasterio.open(tif_path) as src:
                if src.count == required_band_count:
                    valid_files.append(tif_path)
        except Exception:
            continue
    return valid_files

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python filter_valid_tifs.py <raw_dir> [required_band_count]")
        sys.exit(1)
    raw_dir = sys.argv[1]
    required_band_count = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    valid_tifs = filter_valid_tifs(raw_dir, required_band_count)
    print(f"{len(valid_tifs)} fichiers .tif valides trouvés dans {raw_dir} :")
    for path in valid_tifs:
        print(" -", path)
