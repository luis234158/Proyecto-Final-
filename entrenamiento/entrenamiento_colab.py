# Entrenamiento del detector de monedas - Google Colab
# Cada bloque "# %%" es una celda del cuaderno.
# Antes de empezar: Entorno de ejecución -> Cambiar tipo de entorno -> GPU (T4)

# %% 1. Instalar librerías
!pip install ultralytics roboflow

# %% 2. Verificar que la GPU está activa
!nvidia-smi

# %% 3. Descargar el dataset desde Roboflow (formato YOLOv11)
# Este bloque lo genera Roboflow en "Export -> YOLOv11 -> show download code"
from roboflow import Roboflow

rf = Roboflow(api_key="TU_API_KEY")          # no subir la API key real a GitHub
project = rf.workspace("TU_WORKSPACE").project("TU_PROYECTO")
version = project.version(1)
dataset = version.download("yolov11")

# %% 4. Entrenar (transfer learning desde yolo11n preentrenado)
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)

# %% 5. Validar y ver métricas (mAP50, mAP50-95, matriz de confusión)
metrics = model.val()
print(metrics.box.map50, metrics.box.map)

from IPython.display import Image, display
display(Image("runs/detect/train/results.png"))
display(Image("runs/detect/train/confusion_matrix.png"))

# %% 6. Probar el modelo sobre las imágenes de test
best = YOLO("runs/detect/train/weights/best.pt")
best.predict(source=f"{dataset.location}/test/images", save=True, conf=0.5)

# %% 7. Descargar best.pt para usarlo en el PC con detector.py
from google.colab import files
files.download("runs/detect/train/weights/best.pt")
