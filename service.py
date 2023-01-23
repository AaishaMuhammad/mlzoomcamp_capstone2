from flask import Flask
from flask import request
from flask import jsonify
from predict import predict

app = Flask("predict")

@app.route("/predict", methods=["POST"])
def prediction():

    image_file = request.files['img']
    output = predict(image_file)
    result = {"prediction": output}

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="localhost", port=8080)