import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input2
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_predictions2
import numpy as np
from PIL import Image
from io import BytesIO

# model = ResNet50(weights='imagenet')
model = VGG16(weights='imagenet')

def read_image(file) -> Image.Image:
    pil_image = Image.open(BytesIO(file))
    #insert log statement
    return pil_image

def transform(file: Image.Image):

    #img_path = 'image3.jpeg'
    #img = image.load_img(file, target_size=(224, 224))
    img = np.asarray(file.resize((224, 224)))[..., :3]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
    #result = {}
    print('Predicted:', decode_predictions(preds, top=3)[0])
    #result = decode_predictions(preds, top=3)[0]
    result = decode_predictions(model.predict(x), 3)[0]
    
    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"

        response.append(resp)

    return response