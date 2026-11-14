"""Ayuda para que los scripts fallen con un mensaje claro cuando faltan datos (que no vienen en el repo)."""
import os, sys

def exigir(ruta, ayuda):
    """Sale con un mensaje útil si `ruta` no existe. Todos los scripts se corren desde la raíz del repo."""
    if not os.path.exists(ruta):
        sys.exit(f"Falta «{ruta}». {ayuda}\nRecuerda correr los scripts desde la raíz del repositorio (ver README.md).")
