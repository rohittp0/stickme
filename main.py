from typing import List, Tuple

from transformers import pipeline

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools


def get_photos_client():
    SCOPES = 'https://www.googleapis.com/auth/photoslibrary.readonly'

    store = file.Storage('token-for-google.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_id.json', SCOPES)
        creds = tools.run_flow(flow, store)

    return build('photoslibrary', 'v1', http=creds.authorize(Http()))


def get_from_album(album_id: str) -> List[Tuple[str, str]]:
    """
     Get images from Google Photos album
    :param album_id: The id of the album
    :return: A list of image urls and ids
    """


def main():
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    text = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")

    print(text)


if __name__ == '__main__':
    main()
