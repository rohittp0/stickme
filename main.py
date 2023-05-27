import os

import numpy as np
from flask import Flask, request, send_from_directory

from caption import load_captions, get_image_by_id
from search import load_model, get_similarity_index

app = Flask(__name__)

embeddings = np.load("captions.npy")
tokenizer, model = load_model()
captions = load_captions()


# Search route with query parameter
@app.route("/search")
def search():
    query = request.args.get("q")
    if not query:
        return "No query specified", 400

    query = query.lower()
    indices = get_similarity_index(query, tokenizer, model, embeddings)

    results = []
    for index in indices:
        image_id, caption = captions[index]
        results.append({"img": get_image_by_id(image_id), "caption": caption})

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
    app.run(debug=True, host="0.0.0.0", port=5000)
