from PIL import Image
from model_predict import read_image, transform, resultsjson
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

app = FastAPI() #load fastapi


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/files/")
async def create_file(file):
    return {"file_size": len(file)}
    
@app.post("/uploadfile/")
async def main(file):
    # read image
    imagem = read_image(file)
    # transform and get result 
    result = transform(imagem)
    # get prediction
    prediction = resultsjson(result)
    return prediction