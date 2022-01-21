
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route("/")
@cross_origin()
def home_page():
    return render_template("index.html", outcome=None)


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("./"+secure_filename(f.filename))
        return render_template("index.html", outcome=classify_image("./"+secure_filename(f.filename)))


def classify_image(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    model=load_model("./cat_dog_classifier.h5")
    os.remove(image_path)
    return "It's a DOG" if model.predict(test_image)[0][0]==1 else "It's a CAT"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
