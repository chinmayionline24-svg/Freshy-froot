import pickle
import io
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from Orange.data import Table
from Orange.base import Model


app = FastAPI()

# Allow your Netlify website to connect to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 1. LOAD THE ORANGE MODEL
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "fruit_model.pkcls"

try:
    with open(MODEL_PATH, "rb") as f:
        orange_model = pickle.load(f)

    domain = orange_model.domain

except Exception as e:
    raise RuntimeError(f"Could not load Orange model: {e}")


# --------------------------------------------------
# 2. LOAD SQUEEZENET
# --------------------------------------------------

squeezenet = models.squeezenet1_1(pretrained=True)

# Remove the final classification layer
squeezenet.classifier[1] = torch.nn.Identity()

squeezenet.eval()


# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# --------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# --------------------------------------------------
# 4. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# 5. FRUIT PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    try:
        img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Convert image into a tensor
    img_tensor = preprocess(img).unsqueeze(0)

    # Extract SqueezeNet features
    with torch.no_grad():
        embeddings = squeezenet(img_tensor).flatten().numpy()

    # Convert features into a numeric array
    features = np.asarray(
        embeddings,
        dtype=np.float32
    ).reshape(1, -1)

    # Check feature compatibility
    if features.shape[1] != len(domain.attributes):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Feature mismatch: model expects "
                f"{len(domain.attributes)} features, "
                f"but received {features.shape[1]}"
            )
        )

    # Create Orange data table
    row = Table.from_numpy(
        domain,
        X=features
    )

    # Get prediction and probabilities
    values, probabilities = orange_model(
        row,
        ret=Model.ValueProbs
    )

    # Identify predicted class
    class_index = int(values[0])
    prediction_label = domain.class_var.values[class_index]

    # Get confidence
    confidence = float(probabilities[0][class_index])

    # Return result to website
    return {
        "prediction": str(prediction_label),
        "confidence": confidence
    }
