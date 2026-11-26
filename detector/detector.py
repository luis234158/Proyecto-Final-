import time
from collections import Counter

import cv2
import serial
from ultralytics import YOLO

PUERTO = "COM9"
CAMARA = 1
VELOCIDAD = 30
TIEMPO_ENTREGA = 3000
FRAMES_VOTO = 8
ZONA = (0.35, 0.0, 0.65, 1.0)

model = YOLO("best.pt")
print("Clases del modelo:", model.names)
valores = {"50": 50, "100": 100, "200": 200, "500": 500, "1000": 1000}


class ESP32:
    def __init__(self, puerto):
        self.ser = None
        try:
            self.ser = serial.Serial(puerto, 115200, timeout=0.05)
            if not self.esperar("LISTO", 4):
                self.enviar("E")
                self.esperar("E,", 1)
        except serial.SerialException as e:
            print(f"Sin ESP32 ({e}). Los comandos solo se muestran en consola.")

    def enviar(self, cmd):
        print(">", cmd)
        if self.ser:
            self.ser.write((cmd + "\n").encode())

    def esperar(self, prefijo, timeout=2.0):
        if not self.ser:
            return True
        fin = time.time() + timeout
        while time.time() < fin:
            linea = self.ser.readline().decode(errors="ignore").strip()
            if linea:
                print("<", linea)
                if linea.startswith(prefijo):
                    return True
        return False

    def cerrar(self):
        if self.ser:
            self.ser.close()


def detectar(frame):
    h_img, w_img = frame.shape[:2]
    r = model(frame, conf=0.6, iou=0.4, agnostic_nms=True, verbose=False)[0]
    monedas, descartadas = [], []
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        relacion = w / h if h > 0 else 0
        area_rel = (w * h) / (w_img * h_img)
        clase = model.names[int(box.cls)]
        if not (0.75 < relacion < 1.33) or area_rel > 0.08:
            descartadas.append((x1, y1, x2, y2))
            continue
        if clase not in valores:
            clase = "rechazo"
        monedas.append((x1, y1, x2, y2, clase, float(box.conf)))
    return monedas, descartadas


def en_zona(det, zona):
    cx, cy = (det[0] + det[2]) // 2, (det[1] + det[3]) // 2
    return zona[0] <= cx <= zona[2] and zona[1] <= cy <= zona[3]


esp = ESP32(PUERTO)
esp.enviar("B,0")
esp.enviar("A,90")

cap = cv2.VideoCapture(CAMARA, cv2.CAP_DSHOW)
estado = "PAUSA"
votos, frames_voto = [], 0
fin_entrega = 0
conteo = Counter()
total = 0
ultima = ""

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h_img, w_img = frame.shape[:2]
    zona = (int(ZONA[0] * w_img), int(ZONA[1] * h_img),
            int(ZONA[2] * w_img), int(ZONA[3] * h_img))

    monedas, descartadas = detectar(frame)
    en_la_zona = [m for m in monedas if en_zona(m, zona)]

    if estado == "BUSCANDO" and en_la_zona:
        esp.enviar("B,0")
        estado = "CONFIRMANDO"
        votos, frames_voto = [], 0

    elif estado == "CONFIRMANDO":
        frames_voto += 1
        if en_la_zona:
            votos.append(max(en_la_zona, key=lambda m: m[5])[4])
        if frames_voto >= FRAMES_VOTO:
            if len(votos) >= FRAMES_VOTO // 2:
                clase = Counter(votos).most_common(1)[0][0]
                esp.enviar(f"C,{clase}")
                esp.esperar("OK,C")
                esp.enviar(f"T,{VELOCIDAD},{TIEMPO_ENTREGA}")
                fin_entrega = time.time() + TIEMPO_ENTREGA / 1000 + 0.2
                conteo[clase] += 1
                total += valores.get(clase, 0)
                ultima = clase
                estado = "ENTREGANDO"
            else:
                esp.enviar(f"B,{VELOCIDAD}")
                estado = "BUSCANDO"

    elif estado == "ENTREGANDO" and time.time() >= fin_entrega:
        esp.enviar(f"B,{VELOCIDAD}")
        estado = "BUSCANDO"

    cv2.rectangle(frame, zona[:2], zona[2:], (0, 255, 255), 1)
    for x1, y1, x2, y2 in descartadas:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
    for x1, y1, x2, y2, clase, conf in monedas:
        color = (0, 0, 255) if clase == "rechazo" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"${clase} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    n = sum(conteo.values())
    detalle = "  ".join(f"${c}:{conteo[c]}" for c in valores if conteo[c])
    if conteo["rechazo"]:
        detalle += f"  rechazo:{conteo['rechazo']}"
    cv2.putText(frame, f"{estado}  ultima: {ultima}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Clasificadas: {n}  Total: ${total}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, detalle, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"vel {VELOCIDAD}%   p: iniciar/pausar   +/-: velocidad   q: salir", (10, h_img - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Clasificador de monedas", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla in (ord("+"), ord("-")):
        VELOCIDAD = max(20, min(100, VELOCIDAD + (5 if tecla == ord("+") else -5)))
        if estado == "BUSCANDO":
            esp.enviar(f"B,{VELOCIDAD}")
    if tecla == ord("p"):
        if estado == "PAUSA":
            esp.enviar(f"B,{VELOCIDAD}")
            estado = "BUSCANDO"
        else:
            esp.enviar("B,0")
            estado = "PAUSA"

esp.enviar("B,0")
esp.enviar("A,90")
esp.esperar("OK,A")
esp.cerrar()
cap.release()
cv2.destroyAllWindows()
