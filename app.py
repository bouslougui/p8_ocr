import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'uploads'

import keras
from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions



# %%
#Get back the convolutional part of a VGG network trained on ImageNet
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# %%
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 128
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 2
lr = 1e-3
epochs=1

# %%
# Create the model
vgg16 = Sequential()

# Add the vgg convolutional base model
vgg16.add(vgg_conv)

# Add new layers
vgg16.add(Flatten())
vgg16.add(Dense(256, activation='relu'))
vgg16.add(Dropout(0.5))
vgg16.add(Dense(classes_num, activation='softmax'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    img  = load_img(file, target_size=IMAGE_SIZE)
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    probs = vgg16.predict(img)[0]
    output = {'Negative:': probs[0], 'Positive': probs[1]}
    return output

app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(threaded=False)
