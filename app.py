from datetime import date
from pathlib import Path
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from pipeline.predict_pipeline import PredictPipeline

app = FastAPI()


# 1. Define input structure
class SalesInput(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    date: date


# 2. Home endpoint
@app.get("/")
def home():
    return {"message": "Store Sales Forecasting API is running "}


# 3. JSON prediction endpoint
@app.post("/predict_json")
def predict_json(input_data: SalesInput):
    try:
        pipeline = PredictPipeline()

        # convert input → DataFrame
        df = pd.DataFrame([input_data.model_dump()])
        df["date"] = pd.to_datetime(df["date"])

        # unique temporary file per request (safe under concurrency)
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="input_",
            dir=artifacts_dir,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
        df.to_csv(temp_path, index=False)

        try:
            preds = pipeline.predict(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return {
            "prediction": float(preds[0])
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))