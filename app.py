from flask import Flask, request, render_template, jsonify
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()
global graph
graph = tf.compat.v1.get_default_graph()

from helper_functions import build_model, MAX_WORD_COUNT, custom_predict, split_sentence

with sess.as_default():
    with graph.as_default():
        model = build_model(MAX_WORD_COUNT+2) 
        model.load_weights('bert_POS.h5')

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index():
    return render_template('./index.html')

@app.route('/submit', methods = ['POST'])
def submit():
    result = None
    if request.method == 'POST':
        total_string = request.form['string']
        strings = split_sentence(total_string)
        result=[]
        for string in strings:
            prediction = custom_predict(model, string)
            result = result +   prediction
    return jsonify(result = result)

@app.route('/tutorial', methods = ['POST', 'GET'])
def tutorial():
    return render_template('./tutorial.html')

@app.route('/history', methods = ['POST', 'GET'])
def history():
    return render_template('./history.html')

@app.route('/about', methods = ['POST', 'GET'])
def about():
    return render_template('./about.html')
