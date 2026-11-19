"""Interfaz local: video a la izquierda, panel de casillas a la derecha. Solo caras: detector combinado de la cara principal (E15) si están los modelos, y Haar con cabezas inclinadas y perfiles como apoyo.
Uso:  .venv/bin/python src/app_blur.py [--fuente 0|archivo.mp4|foto.jpg] [--puerto 8765]
Abre http://localhost:8765. Sin dependencias extra: servidor http.server + video MJPEG.
Las capturas se guardan en salidas_demo/ (solo local, fuera de git)."""
import argparse, json, os, threading, time, collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import cv2
import numpy as np
cv2.setNumThreads(3)   # HILOS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector_caras import Detector, _nms
from ediciones import editar

# --- parámetros (privacidad: falso negativo peor que falso positivo, D2) ---
VENTANA, MIN_VOTOS = 4, 1   # en DETECCIONES (≈2,5 por segundo): se esconde desde la primera vez que se ve (falso negativo peor, D2)
RETENER = 4                 # detecciones (≈1,6 s) que se sigue escondiendo una cara tras perderla
MARGEN = 0.25               # se agranda la caja un 25 % para cubrir frente y mentón
ANCHO_DET = 960             # frontal y perfil se buscan a este ancho; a 480 se perdían las caras del fondo (E8)
ANCHO_GIROS = 480           # los giros (lo más caro) se prueban solo a este ancho: las cabezas inclinadas cercanas son grandes
FPS_MAX = 20                # tope de cuadros por segundo del video mostrado
HILOS = 3                   # hilos que puede usar OpenCV (para no calentar el equipo)
ANCHO_FUSION = 640          # ancho de trabajo del detector combinado: a 640 px la cara principal se encuentra igual que a 800 y tarda un 37 % menos (E20)
FRONTAL_FUSION = None       # función de detección frontal del detector combinado, o None para usar Haar (se decide al arrancar)
INTERVALO = 0.4             # segundos mínimos entre detecciones (≈ 2,5 por segundo): cada una cuesta ≈ 0,3 s de procesador (E8)

ESTADO = {"esconder": True, "modo": "desenfoque", "bloque": 16, "fuerza": 1.0, "color": "negro", "ruido": 80, "niveles": 3,
          "n_caras": 0, "fps": 0.0, "det_ms": 0, "fuente": "", "error": ""}
LOCK = threading.Lock()
ULTIMO_JPG = None
ULTIMO_FRAME = None
ULTIMO_CRUDO = None; N_CRUDO = 0; CAJAS = []; VCAJAS = 0


class Pista:
    def __init__(self, caja):
        self.caja = np.array(caja, float); self.votos = collections.deque([1], maxlen=VENTANA)
        self.sin_ver = 0; self.confirmada = False

    def centro(self):
        x, y, w, h = self.caja; return x + w / 2, y + h / 2


class Seguimiento:
    """Une detecciones con pistas por cercanía y suaviza con votación en ventana."""
    def __init__(self): self.pistas = []

    def actualizar(self, dets):
        usadas = set()
        for p in self.pistas:
            mejor, md = None, 1e9
            cx, cy = p.centro()
            for j, d in enumerate(dets):
                if j in usadas: continue
                dx, dy = d[0] + d[2] / 2 - cx, d[1] + d[3] / 2 - cy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < md and dist < max(p.caja[2], p.caja[3]) * 0.8: mejor, md = j, dist
            if mejor is None:
                p.votos.append(0); p.sin_ver += 1
            else:
                usadas.add(mejor); p.votos.append(1); p.sin_ver = 0
                p.caja = 0.5 * p.caja + 0.5 * np.array(dets[mejor], float)   # suaviza posición
            if sum(p.votos) >= MIN_VOTOS: p.confirmada = True
        for j, d in enumerate(dets):
            if j not in usadas:
                nueva = Pista(d); nueva.confirmada = sum(nueva.votos) >= MIN_VOTOS; self.pistas.append(nueva)
        self.pistas = [p for p in self.pistas if (p.confirmada and p.sin_ver <= RETENER) or (not p.confirmada and p.sin_ver <= VENTANA)]
        return [p for p in self.pistas if p.confirmada]


def unir(cajas, umbral=0.3):
    """Cajas que se solapan mucho (dos pistas de la misma cara) se unen en una: así no se esconde dos veces el mismo lugar."""
    cajas = [list(c) for c in cajas]; cambio = True
    while cambio:
        cambio = False
        for i in range(len(cajas)):
            for j in range(i + 1, len(cajas)):
                a, b = cajas[i], cajas[j]
                ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
                inter = ix * iy
                if inter and inter / min(a[2] * a[3], b[2] * b[3]) >= umbral + 0.2:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1]); x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    cajas[i] = [x0, y0, x1 - x0, y1 - y0]; del cajas[j]; cambio = True; break
            if cambio: break
    return [tuple(c) for c in cajas]


def aplicar(img, cajas, modo, bloque=16, fuerza=1.0, color="negro", ruido=80, niveles=3):
    forma = "rect" if modo == "pixelado" else "ovalo"   # D7/D10: desenfoque, tapado y ruido van en óvalo; el pixelado, en rectángulo (sus bloques son cuadros)
    for caja in unir(cajas):
        editar(img, caja, modo, MARGEN, bloque, fuerza, color, forma, ruido, niveles)


def hilo_deteccion():
    """Detecta en segundo plano con el último cuadro; el video sigue fluido y usa las últimas cajas confirmadas."""
    global CAJAS, VCAJAS
    alta = Detector(giros=False, perfil=False); perfil = Detector(giros=False, frontal=False); giros = Detector(perfil=False); seg = Seguimiento(); vistos = -1; vuelta = 0
    while True:
        with LOCK: frame, num = ULTIMO_CRUDO, N_CRUDO
        if frame is None or num == vistos: time.sleep(0.01); continue
        vistos = num; t = time.time(); vuelta += 1
        H, W = frame.shape[:2]; esc = min(1.0, ANCHO_DET / W)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(gris, (int(W * esc), int(H * esc)), interpolation=cv2.INTER_AREA)
        if FRONTAL_FUSION is not None:                                          # detector combinado (E15): cara principal, hasta 3 cajas
            ef = min(1.0, ANCHO_FUSION / W); gf = cv2.resize(gris, (int(W * ef), int(H * ef)), interpolation=cv2.INTER_AREA)
            items = [(tuple(int(v / ef) for v in c), 0) for c in FRONTAL_FUSION(gf)]
        else: items = [(tuple(int(v / esc) for v in c), 0) for c in alta(g1)]  # frontal Haar, sin ecualizar (E3)
        solo_respaldo = FRONTAL_FUSION is not None                              # con el detector combinado, giros y perfil de Haar solo si no encontró ninguna cara
        if (not items) if solo_respaldo else (vuelta % 2 == 0 or not items):   # giros: cada 2 vueltas (o solo de respaldo), a baja resolución
            e2 = min(1.0, ANCHO_GIROS / W); g2 = cv2.resize(gris, (int(W * e2), int(H * e2)), interpolation=cv2.INTER_AREA)
            items += [(tuple(int(v / e2) for v in c), 0) for c in giros(g2)]
        if (vuelta % 3 == 0) and not (solo_respaldo and items):                # perfil: cada 3 vueltas (no si ya hay caras y hay detector combinado)
            items += [(tuple(int(v / esc) for v in c), 0) for c in perfil(g1)]
        dets = [c for c, _ in _nms(items)]
        cajas = [tuple(p.caja) for p in seg.actualizar(dets)]
        dt = time.time() - t
        with LOCK: CAJAS = cajas; VCAJAS += 1; ESTADO["det_ms"] = int(dt * 1000)
        time.sleep(max(0.0, INTERVALO - dt))


def mover(prev, cur, cajas, esc):
    """Entre detecciones, las cajas siguen a la cara con flujo óptico (Lucas-Kanade) sobre 16 puntos de cada caja."""
    out = []
    for (x, y, w, h) in cajas:
        xs = np.linspace(x + w * 0.2, x + w * 0.8, 4); ys = np.linspace(y + h * 0.2, y + h * 0.8, 4)
        p0 = np.array([[a * esc, b * esc] for b in ys for a in xs], np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(15, 15), maxLevel=2)
        ok = st.ravel() == 1
        if ok.sum() >= 4:
            d = np.median((p1 - p0)[ok].reshape(-1, 2), axis=0) / esc
            if abs(d[0]) < w and abs(d[1]) < h: x, y = x + d[0], y + d[1]
        out.append((x, y, w, h))
    return out


def bucle(fuente):
    global ULTIMO_JPG, ULTIMO_FRAME, ULTIMO_CRUDO, N_CRUDO
    es_foto = isinstance(fuente, str) and fuente.lower().endswith((".jpg", ".jpeg", ".png"))
    cam = None if es_foto else cv2.VideoCapture(fuente, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY) if fuente == 0 else cv2.VideoCapture(fuente)
    foto = cv2.imread(fuente) if es_foto else None
    if not es_foto and not cam.isOpened():
        ESTADO["error"] = "No se pudo abrir la fuente de video (¿permiso de cámara?)"; return
    threading.Thread(target=hilo_deteccion, daemon=True).start()
    t0 = time.time(); n = 0; prev = None; vivas = []; ver = -1
    while True:
        t_cuadro = time.time()
        if es_foto: frame = foto.copy(); time.sleep(0.05)
        else:
            ok, frame = cam.read()
            if not ok:
                if isinstance(fuente, str): cam.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                time.sleep(0.05); continue
        if fuente == 0: frame = cv2.flip(frame, 1)   # efecto espejo de la cámara
        with LOCK:
            ULTIMO_CRUDO = frame; N_CRUDO += 1; e = dict(ESTADO); cajas, v = list(CAJAS), VCAJAS
        pq = 480 / frame.shape[1]; cur = cv2.cvtColor(cv2.resize(frame, (480, int(frame.shape[0] * pq))), cv2.COLOR_BGR2GRAY)
        if v != ver or prev is None: vivas, ver = cajas, v   # detección nueva: se parte de sus cajas
        else: vivas = mover(prev, cur, vivas, pq)           # si no, las cajas siguen a la cara
        prev = cur; cajas = vivas
        salida = frame.copy()
        if e["esconder"]: aplicar(salida, cajas, e["modo"], e["bloque"], e["fuerza"], e["color"], e["ruido"], e["niveles"])
        ok, jpg = cv2.imencode(".jpg", salida, [cv2.IMWRITE_JPEG_QUALITY, 80])
        n += 1
        with LOCK:
            ULTIMO_JPG = jpg.tobytes(); ULTIMO_FRAME = salida
            ESTADO["n_caras"] = len(cajas)
            if time.time() - t0 >= 1: ESTADO["fps"] = round(n / (time.time() - t0), 1); n = 0; t0 = time.time()
        time.sleep(max(0.0, 1.0 / FPS_MAX - (time.time() - t_cuadro)))


PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Desenfoque de privacidad</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1c1c1a;--mut:#6b6a66;--line:#dcdad3;--acc:#1f6f5c}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--ink:#ecebe6;--mut:#9a988f;--line:#34332f;--acc:#5fc4a8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;padding:16px;max-width:1280px;margin:0 auto}
@media(max-width:800px){main{grid-template-columns:1fr}}
.video{background:#000;border-radius:10px;overflow:hidden;border:1px solid var(--line);aspect-ratio:16/9;display:flex}
.video img{width:100%;height:100%;object-fit:contain}
aside{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;align-self:start}
h1{font-size:17px;margin:0 0 12px}h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:18px 0 8px}
label{display:flex;gap:10px;align-items:center;padding:8px 0;cursor:pointer}input[type=checkbox],input[type=radio]{accent-color:var(--acc);width:18px;height:18px}
.off{opacity:.45;cursor:not-allowed}
.sw input{position:absolute;opacity:0;width:0;height:0}
.pista{width:44px;height:24px;border-radius:12px;background:var(--line);position:relative;flex:none;transition:background .15s}
.pista::after{content:"";position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;background:#fff;transition:left .15s}
.sw input:checked+.pista{background:var(--acc)}.sw input:checked+.pista::after{left:23px}
.sw input:focus-visible+.pista{outline:2px solid var(--acc);outline-offset:2px}
.chips{display:flex;flex-wrap:wrap;gap:4px 12px}.chips label{padding:4px 0}#tam.apagado,#fz.apagado,#tp.apagado,#rz.apagado,#cl.apagado{display:none}#como.apagado{opacity:.4;pointer-events:none}small{color:var(--mut)}
.num{font-size:34px;font-weight:600}button{width:100%;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit;cursor:pointer;margin-top:8px}
button:hover{border-color:var(--acc)}#err{color:#c0392b;margin-top:8px}
</style></head><body><main>
<div class="video"><img src="/video" alt="Video con blur"></div>
<aside>
<h1>Desenfoque de privacidad</h1>
<h2>Esconder</h2>
<label class="sw"><input type="checkbox" id="esconder" checked><span class="pista"></span><span id="txtesc">Sí, esconder</span></label>
<div id="como">
<h2>Cómo esconder</h2>
<select id="modo" style="width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit;margin-bottom:6px">
<option value="desenfoque">Desenfoque (óvalo)</option><option value="pixelado">Pixelado (rectángulo)</option><option value="tapado">Tapado sólido (óvalo)</option><option value="ruido">Ruido (óvalo)</option><option value="colores">Reducción de colores (óvalo)</option></select>
<div id="tam"><label style="display:block"><small>Tamaño del bloque: <b id="bv">16</b> × <b id="bv2">16</b> píxeles de la imagen. Cada bloque se pinta con su color promedio.</small>
<input type="range" id="bloque" min="4" max="64" step="1" value="16" style="width:100%"></label></div>
<div id="fz"><label style="display:block"><small>Intensidad del desenfoque: <b id="fv">1.0</b> (kernel gaussiano relativo al tamaño de la cara)</small>
<input type="range" id="fuerza" min="0.2" max="3" step="0.1" value="1" style="width:100%"></label></div>
<div id="tp"><label style="display:block"><small>Color del tapado</small>
<select id="color" style="width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit"><option value="negro">Negro</option><option value="blanco">Blanco</option><option value="promedio">Color promedio de la cara</option></select></label></div>
<div id="rz"><label style="display:block"><small>Intensidad del ruido: <b id="rv">80</b> %. Al 100 % es ruido puro (no queda nada de la cara); con poco ruido la cara se puede recuperar.</small>
<input type="range" id="ruido" min="10" max="100" step="5" value="80" style="width:100%"></label></div>
<div id="cl"><label style="display:block"><small>Intensidad: <b id="nv">8</b> de 10 (<b id="nc">3</b> colores por canal). Con 10 todo queda del mismo gris; con 1 casi no cambia y la cara sigue a la vista.</small>
<input type="range" id="niveles" min="1" max="10" step="1" value="8" style="width:100%"></label></div>
<h2>Detectado ahora</h2>
<div><span class="num" id="n">0</span> <span>caras</span> · <small id="fps"></small></div>
<button id="cap">Guardar captura</button><small id="msg"></small>
<div id="err"></div>
</aside></main>
<script>
const $=id=>document.getElementById(id);
async function envia(){await fetch('/estado',{method:'POST',body:JSON.stringify({esconder:$('esconder').checked,
modo:$('modo').value,bloque:+$('bloque').value,fuerza:+$('fuerza').value,color:$('color').value,ruido:+$('ruido').value,niveles:11-(+$('niveles').value)})})}
function vista(){const on=$('esconder').checked;$('txtesc').textContent=on?'Sí, esconder':'No, mostrar todo';
$('como').classList.toggle('apagado',!on);$('modo').disabled=!on;const m=$('modo').value;$('tam').classList.toggle('apagado',m!=='pixelado');$('fz').classList.toggle('apagado',m!=='desenfoque');$('tp').classList.toggle('apagado',m!=='tapado');$('rz').classList.toggle('apagado',m!=='ruido');$('cl').classList.toggle('apagado',m!=='colores');$('nv').textContent=$('niveles').value;$('nc').textContent=11-(+$('niveles').value);$('rv').textContent=$('ruido').value;$('bv').textContent=$('bv2').textContent=$('bloque').value;$('fv').textContent=(+$('fuerza').value).toFixed(1)}
['esconder'].forEach(i=>$(i).onchange=()=>{vista();envia()});vista();envia();['modo','color'].forEach(i=>$(i).onchange=()=>{vista();envia()});['bloque','fuerza','ruido','niveles'].forEach(i=>$(i).oninput=()=>{vista();envia()});
$('cap').onclick=async()=>{const r=await (await fetch('/captura',{method:'POST'})).json();$('msg').textContent=' '+(r.ruta||r.error)}
setInterval(async()=>{try{const e=await (await fetch('/estado')).json();$('n').textContent=e.n_caras;$('fps').textContent=e.fps+' cuadros/s · detección '+e.det_ms+' ms';$('err').textContent=e.error}catch(_){}},500);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = PAGINA.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path == "/estado":
            with LOCK: self._json(ESTADO)
        elif self.path == "/video":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=cuadro"); self.end_headers()
            try:
                while True:
                    with LOCK: j = ULTIMO_JPG
                    if j:
                        self.wfile.write(b"--cuadro\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError): pass
        else: self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); cuerpo = self.rfile.read(n) if n else b""
        if self.path == "/estado":
            d = json.loads(cuerpo or b"{}")
            with LOCK:
                for k in ("esconder",):
                    if k in d: ESTADO[k] = bool(d[k])
                if isinstance(d.get("bloque"), int) and 4 <= d["bloque"] <= 64: ESTADO["bloque"] = d["bloque"]
                if isinstance(d.get("fuerza"), (int, float)) and 0.2 <= d["fuerza"] <= 3.0: ESTADO["fuerza"] = float(d["fuerza"])
                if d.get("modo") in ("desenfoque", "pixelado", "tapado", "ruido", "colores"): ESTADO["modo"] = d["modo"]
                if isinstance(d.get("niveles"), int) and 1 <= d["niveles"] <= 10: ESTADO["niveles"] = d["niveles"]
                if d.get("color") in ("negro", "blanco", "promedio"): ESTADO["color"] = d["color"]
                if isinstance(d.get("ruido"), (int, float)) and 10 <= d["ruido"] <= 100: ESTADO["ruido"] = float(d["ruido"])
            self._json({"ok": True})
        elif self.path == "/captura":
            with LOCK: f = None if ULTIMO_FRAME is None else ULTIMO_FRAME.copy()
            if f is None: return self._json({"error": "Todavía no hay imagen"})
            os.makedirs("salidas_demo", exist_ok=True); ruta = time.strftime("salidas_demo/captura-%Y%m%d-%H%M%S.jpg")
            cv2.imwrite(ruta, f); self._json({"ruta": ruta})
        else: self.send_error(404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fuente", default="0"); ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--detector", choices=["fusion", "haar"], default="fusion", help="fusion: detector combinado (E15, hace falta el modelo); haar: el anterior")
    ap.add_argument("--modelo", default="modelos/fusion_rapido.pkl"); ap.add_argument("--umbral", type=float, default=0.1, help="0,1 = modo privacidad (E14); 0,2 = menos falsos positivos")
    a = ap.parse_args(); fuente = int(a.fuente) if a.fuente.isdigit() else a.fuente
    ESTADO["fuente"] = str(fuente)
    if a.detector == "fusion":
        if os.path.exists(a.modelo) and os.path.exists("modelos/hogsvm_cara.npz"):
            from detector_fusion import cargar_detector; FRONTAL_FUSION = cargar_detector(a.umbral, a.modelo); print(f"Detector combinado: {a.modelo}, umbral {a.umbral}", flush=True)
        else: print("AVISO: faltan los modelos del detector combinado (modelos/); uso Haar. Cómo entrenarlos: README.md, sección «Detector de la cara principal».", flush=True)
    else: print("Detector Haar (--detector haar)", flush=True)
    threading.Thread(target=bucle, args=(fuente,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.puerto), H)   # solo local: no se expone a la red
    print(f"Listo en http://localhost:{a.puerto}  (Ctrl+C para salir)", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
