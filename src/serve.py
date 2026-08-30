import sys
from pathlib import Path

# Add the parent directory of 'src' to sys.path to enable executing serve.py directly or via uvicorn
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.dataset import get_transforms
from src.model import get_model

app = FastAPI(title="CIFAR-10 Classifier")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MODEL_PATH = "checkpoints/classifier_v1.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(
    architecture="resnet18",
    num_classes=10,
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = get_transforms(train=False)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()

    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "probabilities": probabilities[0].tolist(),
    }
