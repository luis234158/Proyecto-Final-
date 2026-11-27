# %%
!pip install ultralytics roboflow

# %%
!nvidia-smi

# %%
from roboflow import Roboflow

rf = Roboflow(api_key="TU_API_KEY")
project = rf.workspace("TU_WORKSPACE").project("detector-de-monedas")
version = project.version(2)
dataset = version.download("yolov11")

# %%
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=30,
)

# %%
metrics = model.val()
print(metrics.box.map50, metrics.box.map)

from IPython.display import Image, display
display(Image("runs/detect/train/results.png"))
display(Image("runs/detect/train/confusion_matrix.png"))

# %%
best = YOLO("runs/detect/train/weights/best.pt")
best.predict(source=f"{dataset.location}/test/images", save=True, conf=0.5)

# %%
from google.colab import files
files.download("runs/detect/train/weights/best.pt")
