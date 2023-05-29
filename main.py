from PIL import Image
from model_predict import read_image
from model_predict import transform
from fastapi import FastAPI, File, UploadFile
from io import BytesIO


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/files/")
async def create_file(file: bytes = File(...)):
  
    return {"file_size": len(file)}
    


@app.get("/uploadfile/")
async def main(file: bytes = File(...)):

    # read image
    imagem = read_image(file)
    # transform and prediction 
    prediction = transform(imagem)

    return prediction