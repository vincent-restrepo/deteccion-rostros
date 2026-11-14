"""Descarga datasets de Kaggle con la API (usa el python3 del sistema, no el .venv)."""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
for par in sys.argv[1:]:
    ref, destino = par.split("=")
    print("bajando", ref, "->", destino, flush=True)
    api.dataset_download_files(ref, path=destino, unzip=True, quiet=False)
    print("listo", ref, flush=True)
print("FIN")
