import pickle
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Orange.data import Table

app = FastAPI()

# Allow all websites to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD THE ORANGE MODEL ---
# ⚠️ IMPORTANT: Change "model.pkcls" to the exact name of the file you uploaded!
with open("fruit_model.pkcls", "rb") as f:
    orange_model = pickle.load(f)

# --- 2. REBUILD THE DOMAIN ---
domain = orange_model.domain

# --- 3. LOAD SQUEEZENET ---
squeezenet = models.squeezenet1_1(pretrained=True)
squeezenet.classifier[1] = torch.nn.Identity() 
squeezenet.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Convert image to SqueezeNet embeddings
    img_tensor = preprocess(img).unsqueeze(0) 
    
    with torch.no_grad():
        embeddings = squeezenet(img_tensor).flatten().numpy()

    # Feed embeddings to Orange model
    row = Table.from_numpy(domain, np.array([embeddings], dtype=object))
    prediction = orange_model(row)
    
    # Return the result
    return {
        "prediction": str(prediction[0])
    }
