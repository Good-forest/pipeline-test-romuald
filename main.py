import sys
import logging

sys.path.append('modules')

from gee_downloader import main as gee_main
from sen2sr_processor import main as sen2sr_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    try:
        logging.info("=== [1/2] Téléchargement des images GEE ===")
        gee_main()
        logging.info("=== [2/2] Traitement des images avec SEN2SR ===")
        sen2sr_main()
        logging.info("=== Pipeline complet terminé ===")
    except Exception as e:
        logging.exception(f"Erreur critique dans le pipeline : {e}")
