from fastapi import FastAPI
from pydantic import BaseModel
import src.predict_model as predict_model

app = FastAPI()

class Item(BaseModel):
    data_folder: str

@app.post("/predict")
async def predict(item: Item):
    prediction = predict_model.predict(item.data_folder)
    return {"accuracy": prediction}