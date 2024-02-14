from flask import Flask, render_template, Response, request, redirect, url_for

import cv2

import numpy as np

from tensorflow.keras.models import load_model
global vs
from detect_mask_video import project

app = Flask(__name__)

# Load face detector and mask detector models
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/mask')
def openmask():
   
    project()
    return redirect(url_for('main'))

if __name__ == "__main__":
    app.run(debug=True)













