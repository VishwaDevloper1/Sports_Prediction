from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import numpy as np
app = Flask(__name__)


def predict(img_path):
  class_labels = ['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
 'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding', 'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing',
 'cricket', 'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men', 'figure skating pairs', 'figure skating women', 'fly fishing'
, 'football', 'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw', 'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping'
    , 'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing', 'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse',
 'log rolling', 'luge', 'motorcycle racing']
  model1 = load_model('cnn_model.h5')
  img = image.load_img(img_path, target_size=(299, 299))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = preprocess_input(img_array)

# Make predictions
  predictions = model1.predict(img_array)

  predicted_class_index = np.argmax(predictions, axis=1)[0]
  predicted_class_label = class_labels[predicted_class_index]
  predicted_class_probability = predictions[0][predicted_class_index]

  return (f"{predicted_class_label.capitalize()} and its probability is {predicted_class_probability*100:.2f}%")

# Set the folder where uploaded images will be stored
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index.html');


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    if file.filename == '':
        return 'No selected file'

    if file:
        # Save the image in the UPLOAD_FOLDER
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
           # Return the path of the uploaded image
        prediction = predict(image_path)
        return render_template("results.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=False)

