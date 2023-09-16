import os
from flask import Flask, send_from_directory, request

app = Flask(__name__, static_folder='client/build')
upload_folder = "./uploads/"

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(str(app.static_folder) + '/' + path):
        return send_from_directory(str(app.static_folder), path)
    else:
        return send_from_directory(str(app.static_folder), 'index.html')

@app.route("/api")
def api():
    if 'file' not in request.files:
        return '', 422
    f = request.files['file']
    f.save(os.path.join(upload_folder, f.filename))
    # TODO call model
    return f.filename


if __name__ == "__main__":
    app.run()
