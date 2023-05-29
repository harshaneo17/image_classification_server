# Image Classification Server
A ML API that serves image classification model with FastAPI.
This API serves the ensemble model's result.

USAGE
    
    python3 -m venv env

    source env/bin/activate

    pip install -r requirements.txt

run this command 

    uvicorn main:app --reload

The API can be tested by running the uvicorn command and going to /docs page in the URL bar

example: 
    
    http://127.0.0.1:8000/docs 
    
The two models used in this API are not trained on custom dataset. They use imagenet weights to make predictions and give results. 
Ensemble model uses the average of predictions from both models. The images were pre processed using respective preprocess methods for vgg16 and resnet50 models.

The weights derived from [ImageNet](https://www.image-net.org/index.php) dataset was used to make inference