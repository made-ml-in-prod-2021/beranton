import os
import logging.config
from fastapi import FastAPI, HTTPException
import uvicorn

from typing import List

from utils.predict import predict
from utils.checkers import InputDataModel, OutputDataModel

SRC_DIR = "src"

app = FastAPI()
logger = logging.getLogger("online_inference/predict")


@app.on_event("startup")
def load_transformer():
    global transformer
    filepath = os.path.join(SRC_DIR, "transformer.pkl")
    try:
        transformer = load_transformer(filepath)
        logger.info("Loaded transformer")
    except FileNotFoundError as err:
        logger.error(err)
        return

@app.on_event("startup")
def load_model():
    filepath = os.path.join(SRC_DIR, "model.pkl")
    try:
        transformer = load_model(filepath)
        logger.info("Loaded model")
    except FileNotFoundError as err:
        logger.error(err)
        return


@app.get("/")
def main():

    return "Start page"


@app.get("/health")
def health() -> bool:
    models_status = (transformer is not None) and (model is not None)

    return models_status

@app.get("/predict", response_model=OutputDataModel)
def make_prediction(data: InputDataModel):
    if not health():
        logger.error("Model and transformer are not loaded")
        raise HTTPException(status_code=500, detail="Model and transformer are not loaded")
    
    prediction = predict(data=data.convert_to_pandas(),
                         transformer=transformer,
                         model=model)
    
    return OutputDataModel(label=int(prediction))


if __name__ == "__main__":
    uvicorn.run("prediction:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
