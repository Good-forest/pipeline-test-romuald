from pathlib import Path

# Création de la structure de dossiers
Path('config').mkdir(parents=True, exist_ok=True)
Path('data').mkdir(parents=True, exist_ok=True)
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('data/cadastre').mkdir(parents=True, exist_ok=True)
Path('data/outputs').mkdir(parents=True, exist_ok=True)
Path('modules').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)
Path('pipelines').mkdir(parents=True, exist_ok=True)


