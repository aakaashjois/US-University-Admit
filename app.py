from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dill import load
import numpy as np
import os

app = Flask(__name__)
CORS(app)
model = load(open('./models/rf_v1.pkl', 'rb'))
port = int(os.environ.get('PORT', 5000))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', port=port)
    if request.method == 'POST':
        data = np.array(request.get_json(force=True), dtype=np.float)
        pred = model.predict(data.reshape(1, -1))
        response = jsonify(prediction=pred[0])
        response.status_code = 200
        return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
