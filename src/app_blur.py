"""Interfaz local: video a la izquierda, panel de casillas a la derecha. Hoy solo caras (Haar clásico, con cabezas inclinadas y perfiles).
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

# --- parámetros (privacidad: falso negativo peor que falso positivo, D2) ---
VENTANA, MIN_VOTOS = 9, 5   # votación: la cara se confirma si aparece en >=5 de los últimos 9 cuadros
RETENER = 20                # cuadros que se sigue difuminando una cara confirmada tras perderla
MARGEN = 0.25               # se agranda la caja un 25 % para cubrir frente y mentón
ANCHO_DET = 960             # frontal y perfil se buscan a este ancho; a 480 se perdían las caras del fondo (E8)
ANCHO_GIROS = 480           # los giros (lo más caro) se prueban solo a este ancho: las cabezas inclinadas cercanas son grandes
FPS_MAX = 20                # tope de cuadros por segundo del video mostrado
HILOS = 3                   # hilos que puede usar OpenCV (para no calentar el equipo)
INTERVALO = 0.4             # segundos mínimos entre detecciones (≈ 2,5 por segundo): cada una cuesta ≈ 0,3 s de procesador (E8)

ESTADO = {"esconder": True, "cajas": False, "modo": "desenfoque", "bloque": 16,
          "n_caras": 0, "fps": 0.0, "det_ms": 0, "fuente": "", "error": ""}
LOCK = threading.Lock()
ULTIMO_JPG = None
ULTIMO_FRAME = None
ULTIMO_CRUDO = None; N_CRUDO = 0; CAJAS = []; DETS = []


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
            if j not in usadas: self.pistas.append(Pista(d))
        self.pistas = [p for p in self.pistas if (p.confirmada and p.sin_ver <= RETENER) or (not p.confirmada and p.sin_ver <= VENTANA)]
        return [p for p in self.pistas if p.confirmada]


def pixelar(r, b):
    """Mosaico exacto: bloques de b×b píxeles, cada uno pintado con el promedio de color de sus píxeles."""
    h, w = r.shape[:2]; hh, ww = -(-h // b) * b, -(-w // b) * b
    p = cv2.copyMakeBorder(r, 0, hh - h, 0, ww - w, cv2.BORDER_REPLICATE)
    m = p.reshape(hh // b, b, ww // b, b, p.shape[2]).mean(axis=(1, 3))
    return np.repeat(np.repeat(m, b, 0), b, 1)[:h, :w].astype(r.dtype)


def aplicar(img, cajas, modo, bloque=16):
    H, W = img.shape[:2]
    for x, y, w, h in cajas:
        mx, my = w * MARGEN, h * MARGEN
        x0, y0 = int(max(0, x - mx)), int(max(0, y - my)); x1, y1 = int(min(W, x + w + mx)), int(min(H, y + h + my))
        if modo == "pixelado":   # la cuadrícula parte del (0,0) de la imagen, así los bloques quedan alineados
            x0, y0 = x0 // bloque * bloque, y0 // bloque * bloque
        r = img[y0:y1, x0:x1]
        if r.size == 0: continue
        if modo == "pixelado":
            r2 = pixelar(r, bloque)
        else:
            # el desenfoque se hace sobre la región reducida a la cuarta parte (mismo aspecto, mucho menos cálculo)
            h0, w0 = r.shape[:2]; q = cv2.resize(r, (max(1, w0 // 4), max(1, h0 // 4)), interpolation=cv2.INTER_AREA)
            k = max(7, ((w0 // 8) | 1))
            q = cv2.GaussianBlur(cv2.GaussianBlur(q, (k, k), 0), (k, k), 0)
            r2 = cv2.resize(q, (w0, h0), interpolation=cv2.INTER_LINEAR)
        img[y0:y1, x0:x1] = r2


def hilo_deteccion():
    """Detecta en segundo plano con el último cuadro; el video sigue fluido y usa las últimas cajas confirmadas."""
    global CAJAS, DETS
    alta = Detector(giros=False, perfil=False); perfil = Detector(giros=False, frontal=False); giros = Detector(perfil=False); seg = Seguimiento(); vistos = -1; vuelta = 0
    while True:
        with LOCK: frame, num = ULTIMO_CRUDO, N_CRUDO
        if frame is None or num == vistos: time.sleep(0.01); continue
        vistos = num; t = time.time(); vuelta += 1
        H, W = frame.shape[:2]; esc = min(1.0, ANCHO_DET / W)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(gris, (int(W * esc), int(H * esc)), interpolation=cv2.INTER_AREA)
        items = [(tuple(int(v / esc) for v in c), 0) for c in alta(g1)]      # frontal, sin ecualizar (E3)
        if vuelta % 2 == 0 or not items:                                       # giros: cada 2 vueltas, a baja resolución
            e2 = min(1.0, ANCHO_GIROS / W); g2 = cv2.resize(gris, (int(W * e2), int(H * e2)), interpolation=cv2.INTER_AREA)
            items += [(tuple(int(v / e2) for v in c), 0) for c in giros(g2)]
        if vuelta % 3 == 0:                                                    # perfil: cada 3 vueltas
            items += [(tuple(int(v / esc) for v in c), 0) for c in perfil(g1)]
        dets = [c for c, _ in _nms(items)]
        cajas = [tuple(p.caja) for p in seg.actualizar(dets)]
        dt = time.time() - t
        with LOCK: CAJAS, DETS = cajas, dets; ESTADO["det_ms"] = int(dt * 1000)
        time.sleep(max(0.0, INTERVALO - dt))


def bucle(fuente):
    global ULTIMO_JPG, ULTIMO_FRAME, ULTIMO_CRUDO, N_CRUDO
    es_foto = isinstance(fuente, str) and fuente.lower().endswith((".jpg", ".jpeg", ".png"))
    cam = None if es_foto else cv2.VideoCapture(fuente, cv2.CAP_AVFOUNDATION) if fuente == 0 else cv2.VideoCapture(fuente)
    foto = cv2.imread(fuente) if es_foto else None
    if not es_foto and not cam.isOpened():
        ESTADO["error"] = "No se pudo abrir la fuente de video (¿permiso de cámara?)"; return
    threading.Thread(target=hilo_deteccion, daemon=True).start()
    t0 = time.time(); n = 0
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
            ULTIMO_CRUDO = frame; N_CRUDO += 1; e = dict(ESTADO); cajas, dets = list(CAJAS), list(DETS)
        salida = frame.copy()
        if e["esconder"]: aplicar(salida, cajas, e["modo"], e["bloque"])
        if e["cajas"]:
            for d in dets: cv2.rectangle(salida, (d[0], d[1]), (d[0] + d[2], d[1] + d[3]), (0, 200, 255), 2)
            for c in cajas: cv2.rectangle(salida, (int(c[0]), int(c[1])), (int(c[0] + c[2]), int(c[1] + c[3])), (0, 255, 0), 2)
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
.chips{display:flex;flex-wrap:wrap;gap:4px 12px}.chips label{padding:4px 0}#tam.apagado{opacity:.4;pointer-events:none}#como.apagado{opacity:.4;pointer-events:none}small{color:var(--mut)}
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
<label><input type="radio" name="modo" value="desenfoque" checked> Desenfoque</label>
<label><input type="radio" name="modo" value="pixelado"> Pixelado</label>
<div id="tam"><small>Tamaño del bloque (píxeles de la imagen): cada bloque N×N se pinta con su color promedio</small>
<div class="chips"><label><input type="radio" name="bloque" value="4">4×4</label><label><input type="radio" name="bloque" value="8">8×8</label><label><input type="radio" name="bloque" value="16" checked>16×16</label><label><input type="radio" name="bloque" value="32">32×32</label><label><input type="radio" name="bloque" value="64">64×64</label></div></div>
</div>
<h2>Depuración</h2>
<label><input type="checkbox" id="cajas"> Mostrar cajas <small>(naranja: detección; verde: confirmada)</small></label>
<h2>Detectado ahora</h2>
<div><span class="num" id="n">0</span> <span>caras</span> · <small id="fps"></small></div>
<button id="cap">Guardar captura</button><small id="msg"></small>
<div id="err"></div>
</aside></main>
<script>
const $=id=>document.getElementById(id);
async function envia(){await fetch('/estado',{method:'POST',body:JSON.stringify({esconder:$('esconder').checked,cajas:$('cajas').checked,
modo:document.querySelector('input[name=modo]:checked').value,bloque:+document.querySelector('input[name=bloque]:checked').value})})}
function vista(){const on=$('esconder').checked;$('txtesc').textContent=on?'Sí, esconder':'No, mostrar todo';
$('como').classList.toggle('apagado',!on);document.querySelectorAll('input[name=modo]').forEach(r=>r.disabled=!on);$('tam').classList.toggle('apagado',document.querySelector('input[name=modo]:checked').value!=='pixelado')}
['esconder','cajas'].forEach(i=>$(i).onchange=()=>{vista();envia()});vista();document.querySelectorAll('input[name=modo],input[name=bloque]').forEach(r=>r.onchange=()=>{vista();envia()});
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
                for k in ("esconder", "cajas"):
                    if k in d: ESTADO[k] = bool(d[k])
                if d.get("bloque") in (4, 8, 16, 32, 64): ESTADO["bloque"] = d["bloque"]
                if d.get("modo") in ("desenfoque", "pixelado"): ESTADO["modo"] = d["modo"]
            self._json({"ok": True})
        elif self.path == "/captura":
            with LOCK: f = None if ULTIMO_FRAME is None else ULTIMO_FRAME.copy()
            if f is None: return self._json({"error": "Todavía no hay imagen"})
            os.makedirs("salidas_demo", exist_ok=True); ruta = time.strftime("salidas_demo/captura-%Y%m%d-%H%M%S.jpg")
            cv2.imwrite(ruta, f); self._json({"ruta": ruta})
        else: self.send_error(404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fuente", default="0"); ap.add_argument("--puerto", type=int, default=8765)
    a = ap.parse_args(); fuente = int(a.fuente) if a.fuente.isdigit() else a.fuente
    ESTADO["fuente"] = str(fuente)
    threading.Thread(target=bucle, args=(fuente,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.puerto), H)   # solo local: no se expone a la red
    print(f"Listo en http://localhost:{a.puerto}  (Ctrl+C para salir)", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
