from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from utils.preprocess import preprocess_text  # Your preprocessing logic

app = Flask(__name__)
model = load_model('model/rnn_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    processed_input = preprocess_text(email_text)  # tokenize and pad
    prediction = model.predict(processed_input)
    label = "Spam" if prediction[0][0] > 0.5 else "Not Spam"
    return render_template('index.html', prediction=label, email=email_text)

if __name__ == '__main__':
    app.run(debug=True)
