from flask import Flask, render_template, request
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("svm_cats_dogs_model.joblib")

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64,64))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()/255.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h = cv2.calcHist([hsv],[0],None,[32],[0,180]).flatten()
    s = cv2.calcHist([hsv],[1],None,[32],[0,256]).flatten()
    v = cv2.calcHist([hsv],[2],None,[32],[0,256]).flatten()

    color = np.concatenate([h,s,v])
    color = color/(color.sum()+1e-7)

    return np.concatenate([gray,color]).reshape(1,-1)

@app.route("/", methods=["GET","POST"])
def home():

    prediction = ""

    if request.method == "POST":

        file = request.files["image"]

        path = os.path.join("uploads", file.filename)
        file.save(path)

        features = extract_features(path)

        pred = model.predict(features)[0]

        prediction = "Cat 🐱" if pred == 0 else "Dog 🐶"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)