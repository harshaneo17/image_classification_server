import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from tensorflow.keras.applications.vgg16 import vgg16_preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_predictions2
import numpy as np
from PIL import Image
from io import BytesIO
import logging as log

# sets the basic logging level as info
log.basicConfig(level=log.INFO)

# load pretrained vgg16 and resnet50 models with imagenet weights
vgg16_model = VGG16(weights='imagenet')
resnet50_model = ResNet50(weights='imagenet')


def read_image(file) -> Image.Image:
    """ This function takes file and converts it into PIL image format """
    pil_image = Image.open(BytesIO(file))
    log.info('_____LOADING IMAGE______')
    return pil_image


def transform(file) -> Image.Image:
    """ This function takes the converted image and makes prediction on that after preprocessing it.
    Returns the decoded prediction of ensemble algorithms """

    # loads the image in standard
    img = np.asinput(file.resize((224, 224)))[..., :3]
    vgg16_image_input = image.img_to_input(img)
    vgg16_image_input = np.expand_dims(vgg16_image_input, axis=0)

    # Copy same image for use with resnet50 model
    resnet50_image_input = vgg16_image_input.copy()

    # uses different preprocess methods for each model
    vgg16_image_input = vgg16_preprocess_input(vgg16_image_input)
    resnet50_image_input = resnet50_preprocess_input(resnet50_image_input)

    vgg16_prediction = vgg16_model.predict(vgg16_image_input)
    resnet50_prediction = resnet50_model.predict(resnet50_image_input)

    # ensemble methods with average of two predictions
    averaged_predictions = np.mean([vgg16_prediction, resnet50_prediction], axis=0)

    # RS - Is decode_predictions2 not needed here? Is it just the decode_predictions from vgg16 that's needed?
    log.info(f'Predicted:{decode_predictions(averaged_predictions, top=1)[0]}')
    result = decode_predictions(averaged_predictions, 1)[0]

    return result


def resultsjson(listobject) -> list:
    """ This function takes listobject from the above function and makes it a response object.
    Returns response object """
    response = []
    for i, res in enumerate(listobject):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        response.append(resp)

    return response