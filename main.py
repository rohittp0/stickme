import os

import numpy as np

from waitress import serve
from flask import Flask, request, send_from_directory
from dotenv import load_dotenv

from caption import load_captions
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
        results.append({"img": image_id, "caption": caption})

    return results


@app.route('/', defaults={'path': None})
@app.route('/<path:path>')
def public(path: str):
    if path is None:
        path = "index.html"
    elif path.find(".") == -1:
        path = "index.html"

    return send_from_directory(f'{os.getcwd()}/public', path)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5555)
