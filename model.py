import shutil

import tensorflow as tf
from tensorflow import keras
import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.get('/')
def home():
    return "THIS IS THE HOME PAGE OF THE API"

#
# @app.post('/{predict}')
# def predict(predict: str):
#     return "NOTHING"


@app.post("/files/")
async def create_file(file: bytes = File(default=None)):
    if not file:
        return {"message": "No file sent"}
    else:
        return {"file_content": file}



@app.post("/predit/")
async def predict(file: UploadFile):
    destination = "images/temp.jpg"
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    model = keras.models.load_model('mask_mobile_net.h5')

    img = tf.keras.utils.load_img(
        'images/temp.jpg', target_size=(224, 224)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    pred_res = predictions.argmax(axis=1)[0]

    # print(predictions[0][0])
    print(predictions)
    print(pred_res)

    if pred_res == 1:
        return {"class": "NO_MASK"}
    elif pred_res == 0:
        return {"class": "MASK_ON"}


# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(predictions[np.argmax(score)], 100 * np.max(score))
# )
