import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input2
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_predictions2
import numpy as np
from PIL import Image
from io import BytesIO
import logging as log

log.basicConfig(level=log.INFO) #sets the basic logging level as info


model = VGG16(weights='imagenet')   #load pretrained vgg16 model with imagenet weights
model2 = ResNet50(weights='imagenet') #load pretrained resnet50 model with imagenet weights


def read_image(file) -> Image.Image:
    '''This function takes file and converts it into PIL image format'''
    pil_image = Image.open(BytesIO(file))
    log.info('_____LOADING IMAGE______')
    return pil_image

def transform(file) -> Image.Image:
    '''This function takes the converted image and makes prediction on that after preprocessing it. 
    Returns the decoded prediction of ensemble algorithms '''
    img = np.asarray(file.resize((224, 224)))[..., :3] #loads the image in standard 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    y = x.copy() 
    x , y = preprocess_input(x),preprocess_input2(y) #uses different preprocess methods for each model
    preds = model.predict(x)
    preds2 = model2.predict(y)
    averaged_predictions = np.mean([preds, preds2], axis=0) #ensemble methods with average of two predictions

    log.info(f'Predicted:{decode_predictions(averaged_predictions, top=1)[0]}')
    result = decode_predictions(averaged_predictions, 1)[0]
    return result
    
def resultsjson(listobject) -> list:
    '''This function takes listobject from the above function and makes it a response object.
    Returns response object'''
    response = []
    for i, res in enumerate(listobject):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        response.append(resp)
    return response