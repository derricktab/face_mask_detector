import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

# checkpoint_path = "training_gender/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

model = models.load_model('mymodel.h5')
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    #Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    #Resizing into dimensions we used while training
    im = im.resize((150,150))
    img_array = np.array(im)

    #Expand dimensions to match the 4D Tensor shape.
    img_array = np.expand_dims(img_array, axis=0)

    #Calling the predict function using keras
    prediction = model.predict(img_array)#[0][0]
    print(prediction[0][0])

    #Customize this part to your liking...
    if(prediction[0][0] == 0):
        print("NO FACE MASK")
    elif(prediction == 1):
        print("FACE MASK ON")


    cv2.imshow("Prediction", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
            break
video.release()
cv2.destroyAllWindows()