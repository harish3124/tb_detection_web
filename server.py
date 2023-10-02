import os, json
from flask import Flask, send_from_directory, request

import model
from model.accuracy import acc

app = Flask(__name__, static_folder="client/build")


# Serve React App
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(str(app.static_folder) + "/" + path):
        return send_from_directory(str(app.static_folder), path)
    else:
        return send_from_directory(str(app.static_folder), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "", 422
    f = request.files["file"]
    result = model.predicto(f)
    return json.dumps({"result": result})


@app.route("/models", methods=["POST"])
def models():
    return acc()


if __name__ == "__main__":
    app.run()
