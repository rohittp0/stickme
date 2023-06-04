import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model


def precompute_embeddings(sentences, save_path):
    tokenizer, model = load_model()

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Get the BERT embeddings for the sentences
    with torch.no_grad():
        embeddings = model(**encoded_input).last_hidden_state

    # Calculate the mean of the embeddings for each sentence
    embeddings_mean = torch.mean(embeddings, dim=1)

    # Save the embeddings to disk
    np.save(save_path, embeddings_mean)


def get_similarity_index(search_query, tokenizer, model, embeddings) -> np.ndarray:
    # Encode the search query with BERT tokenizer and model
    encoded_input = tokenizer(search_query, return_tensors="pt")
    with torch.no_grad():
        search_embedding = model(**encoded_input).last_hidden_state.mean(dim=1)

    # Calculate the cosine similarity between the search query and all the sentences
    similarity = cosine_similarity(search_embedding, embeddings)

    # Find the indices of the most similar sentences
    return np.argsort(-similarity[0])[:5]


def main():
    # captions = load_captions()
    # precompute_embeddings(captions, "captions.npy")

    embeddings = np.load("captions.npy")
    tokenizer, model = load_model()


if __name__ == "__main__":
    main()
