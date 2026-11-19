"""Registra y verifica los SHA-256 de los datos crudos (entradas/*). Por dataset guarda: número de archivos, bytes totales y un hash
de conjunto = SHA-256 de las líneas «ruta_relativa sha256_del_archivo» ordenadas. Así, dos personas pueden comprobar que descargaron lo mismo.
Además guarda el SHA-256 individual de los archivos de anotaciones (los que definen las cajas).
Salida: datos/hashes_datos.json (versionado).
Uso:  .venv/bin/python src/hashes_datos.py generar        # calcula y escribe
      .venv/bin/python src/hashes_datos.py verificar      # recalcula y compara con lo versionado"""
import hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rutas import exigir
DATASETS = {
    "wider-yolo (lylmsc/wider-face-for-yolo-training)": ("entradas/wider-yolo", ()),
    "wider-mksaad (mksaad/wider-face-a-face-detection-benchmark)": ("entradas/wider-mksaad", ("wider_face_split/wider_face_split/wider_face_train_bbx_gt.txt", "wider_face_split/wider_face_split/wider_face_val_bbx_gt.txt")),
    "fddb (faizanarif15/dataset-fddb)": ("entradas/fddb", ("Dataset_FDDB/label.txt",)),
    "openimages (fareselmenshawii/face-detection-dataset)": ("entradas/faces-openimages", ()),
    "keypoints-fkd (drgilermo/face-images-with-marked-landmark-points)": ("entradas/keypoints-fkd", ("facial_keypoints.csv",)),
}
RUTA_JSON = "datos/hashes_datos.json"
IGNORAR = {".DS_Store"}

def sha(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def archivos(raiz):
    out = []
    for d, _, fs in os.walk(raiz):
        for f in fs:
            if f not in IGNORAR: out.append(os.path.join(d, f))
    return sorted(out)

def calcular(solo=None):
    res = {}
    with ProcessPoolExecutor() as ex:
        for nombre, (raiz, anot) in DATASETS.items():
            if solo is not None and nombre not in solo: continue
            exigir(raiz, "Descarga el dataset: ver README.md / AGENTS.md.")
            fs = archivos(raiz); hs = list(ex.map(sha, fs, chunksize=32))
            rel = [os.path.relpath(f, raiz) for f in fs]
            conjunto = hashlib.sha256("".join(f"{r} {h}\n" for r, h in zip(rel, hs)).encode()).hexdigest()
            por = dict(zip(rel, hs))
            res[nombre] = {"carpeta": raiz, "archivos": len(fs), "bytes": sum(os.path.getsize(f) for f in fs),
                           "sha256_conjunto": conjunto, "anotaciones": {a: por.get(a, "NO ENCONTRADO") for a in anot}}
            print(nombre, res[nombre]["archivos"], "archivos", flush=True)
    return res

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo == "generar":
        os.makedirs("datos", exist_ok=True); json.dump(calcular(), open(RUTA_JSON, "w"), indent=1, ensure_ascii=False); print("Escrito", RUTA_JSON)
    elif modo == "agregar":   # calcula solo los datasets que aún no están en el JSON
        exigir(RUTA_JSON, "Primero: generar."); d = json.load(open(RUTA_JSON)); d.update(calcular({n for n in DATASETS if n not in d})); json.dump(d, open(RUTA_JSON, "w"), indent=1, ensure_ascii=False); print("Actualizado", RUTA_JSON)
    elif modo == "verificar":
        exigir(RUTA_JSON, "Primero: python src/hashes_datos.py generar (o trae el archivo del repo).")
        esperado, ahora = json.load(open(RUTA_JSON)), calcular(); mal = 0
        for n, e in esperado.items():
            a = ahora.get(n, {}); ok = a.get("sha256_conjunto") == e["sha256_conjunto"]; mal += not ok
            print(("OK  " if ok else "DIFIERE  ") + n + ("" if ok else f" (archivos {a.get('archivos')} vs {e['archivos']})"))
        sys.exit(1 if mal else 0)
    else: sys.exit("Uso: python src/hashes_datos.py generar|agregar|verificar")
