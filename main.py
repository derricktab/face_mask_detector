import shutil

import PIL.Image
import cv2
import mediapipe as mp
# from mediapipe.python.solutions.drawing_utils import *
import tensorflow as tf
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
model = tf.keras.models.load_model('mask_mobile_net.h5')
# print(model.summary())
FONT = cv2.FONT_HERSHEY_SIMPLEX

# creating an instance of the FastAPI class
app = FastAPI()


@app.get("/")
def home():
    return "FACE MASK DETECTOR HOME PAGE"


@app.post("/image/")
def predimage(received_image: UploadFile):
    # Saving the received image to disk
    destination = "temp.jpg"
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(received_image.file, buffer)
    finally:
        received_image.file.close()

    # For static images:
    IMAGE_FILES = [destination]

    with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7,
    ) as face_detection:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # getting image height
            iheight = image.shape[0]
            print("IMAGE HEIGHT: " + str(iheight))

            # Gettign image width
            iwidth = image.shape[1]
            print("IMAGE WIDTH: " + str(iwidth))

            if results.detections:

                for detection in results.detections:
                    # points for the bouding box of the detected face
                    x = int(detection.location_data.relative_bounding_box.xmin * iwidth)
                    y = int(detection.location_data.relative_bounding_box.ymin * iheight)
                    w = int(detection.location_data.relative_bounding_box.width * iwidth)
                    h = int(detection.location_data.relative_bounding_box.height * iheight)

                    # declaring the face dimensions
                    face = image[y:y + h, x:x + w, :]
                    face = cv2.resize(image, (224, 224))
                    face = tf.keras.preprocessing.image.img_to_array(face)
                    face = face / 255.
                    face = tf.expand_dims(face, axis=0)

                    pred = model.predict(face)
                    pred_res = pred.argmax(axis=1)[0]

                    print(pred)
                    print(pred_res)

                    label = "MASK ON" if pred_res == 0 else "NO MASK"
                    color = (0, 255, 0) if label == "MASK ON" else (0, 0, 255)

                    cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=2)
                    cv2.putText(image, label, (x, y - 10), FONT, 0.5, color, 2)

            annotated_image = image.copy()
            cv2.imwrite('output/outputimage.png', annotated_image)
            return FileResponse("output/outputimage.png")
        else:
            return "NO FACE DETECTED IN IMAGE"

        # # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        # results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #
        # # Draw face detections of each face.
        # if not results.detections:
        #     continue
        #
        # annotated_image = image.copy()
        #
        # for detection in results.detections:
        #     print('Nose tip:')
        #     print(mp_face_detection.get_key_point(
        #         detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
        #     mp_drawing.draw_detection(annotated_image, detection)
        # cv2.imwrite('output/annotated_image.png', annotated_image)
#
# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(image)
#
#     # Draw the face detection annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     if results.detections:
#
#       for detection in results.detections:
#
#         # points for the bouding box of the detected face
#         x = int(detection.location_data.relative_bounding_box.xmin * 640)
#         y = int(detection.location_data.relative_bounding_box.ymin * 480)
#         w = int(detection.location_data.relative_bounding_box.width * 640)
#         h = int(detection.location_data.relative_bounding_box.height * 480)
#
#         # declaring the face dimensions
#         face = image[y:y + h, x:x + w, :]
#         face = cv2.resize(image, (224, 224))
#         face = tf.keras.preprocessing.image.img_to_array(face)
#         face = face / 255.
#         face = tf.expand_dims(face, axis=0)
#
#         pred = model.predict(face)
#         pred_res = pred.argmax(axis=1)[0]
#
#         print(pred)
#         print(pred_res)
#
#         label = "WITH MASK" if pred_res == 0 else "NO MASK"
#         color = (0, 255, 0) if label == "WITH MASK" else (0, 0, 255)
#
#         # label = "{}: {:.2f}%".format(label, max(pred) * 100)
#
#         cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=2)
#         cv2.putText(image, label, (x, y - 10), FONT, 0.5, color, 2)
#
#
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('Face Mask Detector', image)
#
#
#
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()
