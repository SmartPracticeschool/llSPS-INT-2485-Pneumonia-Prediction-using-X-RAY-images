from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions


from keras.models import load_model
from keras import backend
from tensorflow.keras import backend

import tensorflow as tf

graph = tf.get_default_graph()

from skimage.transform import resize
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow.python.keras


app = Flask(__name__)


PATH = 'models/my_model.h5'
model = tensorflow.keras.models.load_model(PATH)

@app.route('/', methods = ['GET'])
def index():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	global graph
	with graph.as_default():
		if request.method == 'POST':
			f = request.files['file']
			#basepath = os.path.dirname("D:\\Machine Learning Projects\\SmartInternz\\Flask Pneumonia Prediction")
			#file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
			file_path = os.path.join("D:\\Machine Learning Projects\\SmartInternz\\Flask Pneumonia Prediction\\uploads", secure_filename(f.filename))
			f.save(file_path)
			img = image.load_img(file_path, target_size = (600, 600))
			x = image.img_to_array(img)
			y = np.expand_dims(x, axis = 0)
			preds = model.predict_classes(y)
			index = ['Normal', 'Pneumonia']
			text = index[preds[0][0]]
			print(text)
			return text

if __name__ == '__main__':	
	app.run(debug = False, threaded = False)




