from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import predict_image

app = FastAPI(
    title="Fruit Freshness Detection API",
    description="Deep Learning API using TensorFlow and FastAPI",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Fruit Freshness Detection API",
        "status": "Running"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    result = await predict_image(file)

    return JSONResponse(result)