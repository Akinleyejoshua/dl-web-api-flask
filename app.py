from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS, cross_origin
import numpy as np
import os
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CORS(app, support_credentials=False)
app.config["CORS_HEADERS"] = "Content-Type"

img_size = 24
channel = 1
unique = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def process_image(img_path):
    img = tf.constant(img_path)
    img = tf.image.decode_jpeg(img, channels=channel)
    img = tf.image.resize(img, size=[img_size, img_size])
    return img

def load_model(path):
    return tf.compat.v2.keras.models.load_model(path)

model = load_model("./models/facial-expression-v1/saved_model_3")

def predict(img_arr):
    img = process_image(img_arr)
    # plt.imsave("./test.jpeg", img)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction)
    label = unique[np.argmax(prediction)]
    print(f"Prediction - {label} score - {np.max(score[0])}")
    return label, np.max(score[0]), np.argmax(prediction)

@app.route("/api/v1/facial-expression/predict", methods=["POST", "GET"])
@cross_origin(allow_headers=["Content-Type"])
def facial_expression_decoder():
    response = request.get_json()
    if response == None:
        return jsonify({"msg": ""})
    else:
        data_str = response["image"]
        point = data_str.find(",")
        base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"

        image = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image))

        if img.mode != "RGB":
            img = img.convert("RGB")

        image_np = np.array(img)
    
        try:
            label, score, val = predict(image_np)
            return jsonify({"label": f"{label}", "score": f"{score}", "val": f"{val}"})
        except Exception as e:
            return jsonify({"msg": f"{e}",})


@app.route("/")
def index():
    return jsonify({"msg": "API Working"})


@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
