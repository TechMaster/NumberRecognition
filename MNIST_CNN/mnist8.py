import base64
import io
import json

import cv2
import numpy as np
import tensorflow as tf
from imageio import imread
from sanic import Sanic
from sanic import response
from sanic_jinja2 import SanicJinja2  # pip install sanic_jinja2

app = Sanic()
jinja = SanicJinja2(app)

modelCNN = None  # Convolution Neural Network Model
modelBPN = None  # Back Propagation Neural Network Model

# Serves files from the static folder to the URL /static
app.static('/js', './js')


@app.route('/')
@jinja.template('index.html')  # decorator method is static method
async def index(request):
    return


@app.route("/upload", methods=['POST'])
async def on_file_upload(request):
    base64str = base64.b64decode(request.form['imgBase64'][0].split(",")[1])
    image = load_image_from_base64(base64str)

    predict_bpn = modelBPN.predict(image.reshape(image.shape[0], 28 * 28))
    number_bpn = np.argmax(predict_bpn[0])

    predict_cnn = modelCNN.predict(image.reshape(image.shape[0], 28, 28, 1))
    number_cnn = np.argmax(predict_cnn[0])

    result = {
        "number_cnn": int(number_cnn),
        "confident_cnn": float(predict_cnn[0][number_cnn]),

        "number_bpn": int(number_bpn),
        "confident_bpn": float(predict_bpn[0][number_bpn]),
    }
    return response.json(json.dumps(result))


def load_image_from_base64(base64str):
    raw_img = imread(io.BytesIO(base64str))
    gray_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray_image, (28, 28), cv2.INTER_NEAREST)
    # cv2.imwrite("number.png", image)
    input = np.asarray([image])  # reconstruct image as an numpy array

    float_input = input.astype('float32') / 255
    return float_input


def load_model_cnn():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    modelCNN = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    modelCNN.load_weights("model.h5")

    modelCNN.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    return modelCNN


def load_model_bpn():
    # load json and create model
    json_file = open('../MNIST_BPN/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    modelBPN = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    modelBPN.load_weights("../MNIST_BPN/model.h5")

    modelBPN.compile(optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return modelBPN


if __name__ == "__main__":
    modelCNN = load_model_cnn()

    modelBPN = load_model_bpn()

    app.run(host="0.0.0.0", port=8000)

'''
Bước 8: Sẽ dùng 2 mạng CNN và BPN để kiểm tra tính chính xác
'''
