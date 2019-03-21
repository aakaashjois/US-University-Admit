from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dill import load
import numpy as np

app = Flask(__name__)
CORS(app)
model = load(open('./models/rf_v1.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def get_prediction():
    data = np.array(request.get_json(force=True), dtype=np.float)
    pred = model.predict(data.reshape(1, -1))
    response = jsonify(prediction=pred[0])
    response.status_code = 200
    return response


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
