import os

import numpy as np
import requests

from waitress import serve
from flask import Flask, request, send_from_directory, send_file
from dotenv import load_dotenv

from caption import load_captions, get_image_by_id
from search import load_model, get_similarity_index

load_dotenv()

app = Flask(__name__)

embeddings = np.load("captions.npy")
tokenizer, model = load_model()
captions = load_captions()


# Search route with query parameter
@app.route("/search")
def search():
    if not request.headers.get("Authorization"):
        return "No authorization header", 401
    if request.headers.get("Authorization") != os.environ.get("AUTHORIZATION"):
        return "Invalid authorization header", 401

    query = request.args.get("q")
    if not query:
        return "No query specified", 400

    query = query.lower()
    indices = get_similarity_index(query, tokenizer, model, embeddings)

    results = []
    for index in indices:
        image_id, caption = captions[index]
        results.append({"img": f"/image/{image_id}", "caption": caption})

    return results


@app.route("/image/<image_id>")
def image(image_id: str):
    # Check header for authorization
    if not request.headers.get("Authorization"):
        return "No authorization header", 401
    if request.headers.get("Authorization") != os.environ.get("AUTHORIZATION"):
        return "Invalid authorization header", 401

    image_id = image_id.removesuffix(".png")

    if os.path.exists(f"cache/img/{image_id}.png"):
        return send_file(image_id, mimetype='image/png')

    image_url = get_image_by_id(image_id)
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"{image_url} : Request failed with status code {response.status_code}, {response.text}")

    with open(f"cache/img/{image_id}.png", "wb") as f:
        f.write(response.content)

    return send_file(f"cache/img/{image_id}.png", mimetype='image/png')


@app.route('/', defaults={'path': None})
@app.route('/<path:path>')
def public(path: str):
    if path is None:
        path = "index.html"
    elif path.find(".") == -1:
        path = "index.html"

    return send_from_directory(f'{os.getcwd()}/public', path)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
