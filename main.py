import csv
import os
from typing import List, Tuple

from transformers import pipeline

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from tqdm import tqdm


def get_cred(scopes) -> Credentials:
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', scopes)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', scopes)
            creds = flow.run_local_server(port=8000)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds


def get_from_photos(save=True) -> List[Tuple[str, str]]:
    """
     Get images from Google Photos
    :return: A list of image urls and ids
    """
    SCOPES = 'https://www.googleapis.com/auth/photoslibrary.readonly'
    creds = get_cred(SCOPES)
    service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)

    data = []
    nextPageToken = None

    while True:
        results = service.mediaItems().search(body={
            "filters": {
                "mediaTypeFilter": {
                    "mediaTypes": ["PHOTO"]
                }
            },
            "pageSize": 100,
            "pageToken": nextPageToken
        }).execute()

        items = results.get('mediaItems', [])
        items = [(item['baseUrl'], item['id']) for item in items]

        # Append to file csv file
        if save:
            with open('images.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerows(items)

        data.extend(items)
        nextPageToken = results.get('nextPageToken', None)
        if nextPageToken is None:
            break

    return data


def image_to_text(image_url: str) -> List[dict]:
    """
    Convert image to text
    :param image_url: The url of the image
    :return: The text in the image
    """
    im2txt = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    text = im2txt(image_url)

    return text


def main():
    # images = get_from_photos()

    images = set()
    with open('images.csv', 'r') as f:
        for row in tqdm(csv.reader(f), desc='Reading images'):
            if len(row):
                images.add((row[0], row[1]))

    # Unique images only
    print(f'Number of unique images: {len(images)}')

    with open('captions.csv', 'w') as f:
        writer = csv.writer(f)
        for image in tqdm(images, desc='Captioning images'):
            image_url, image_id = image
            text = image_to_text(image_url)[0]['generated_text']
            writer.writerow([image_id, text, image_url])


if __name__ == '__main__':
    main()
