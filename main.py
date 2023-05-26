import csv
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Tuple, Generator

import requests
import torch
from PIL import Image
from google.auth.exceptions import RefreshError
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 32
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


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
            try:
                creds.refresh(Request())
            except Exception as e:
                os.unlink('token.json')
                return get_cred(scopes)
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', scopes)
            creds = flow.run_local_server(port=8000)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds


def get_from_photos() -> Generator[Tuple[str, str], None, None]:
    """
     Get images from Google Photos
    :return: A list of image urls and ids
    """
    SCOPES = 'https://www.googleapis.com/auth/photoslibrary.readonly'
    creds = get_cred(SCOPES)
    service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)

    nextPageToken = None

    while True:
        try:
            results = service.mediaItems().search(body={
                "filters": {
                    "mediaTypeFilter": {
                        "mediaTypes": ["PHOTO"]
                    }
                },
                "pageSize": 100,
                "pageToken": nextPageToken
            }).execute()
        except RefreshError:
            creds = get_cred(SCOPES)
            service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)
            continue

        items = results.get('mediaItems', [])
        items = [(item['baseUrl'], item['id']) for item in items]

        yield items

        nextPageToken = results.get('nextPageToken', None)
        if nextPageToken is None:
            break


def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"{image_url} : Request failed with status code {response.status_code}, {response.text}")

    img = Image.open(BytesIO(response.content))
    return img


def predict_step(images_raw: List[Image.Image]) -> List[str]:
    images = []
    for i_image in images_raw:
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    return [txt.strip() for txt in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]


def image_to_text(image_urls: List[str]) -> List[str]:
    """
    Convert image to text
    :param image_urls: The urls of the images
    :return: The text in the image
    """
    images = []
    predictions = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(download_image, image_urls)
        for result in results:
            images.append(result)

            if len(images) >= 25:
                predictions.extend(predict_step(images))
                images = []

    if len(images) > 0:
        predictions.extend(predict_step(images))

    return predictions


def main():
    total_skipped = 0
    total_processed = 0

    with open('captions.csv', 'r+') as f:
        reader = csv.reader(f)
        existing_ids = set([row[0] for row in reader if len(row) > 0])
        existing_count = len(existing_ids)

        print(f"Existing ids: {existing_count}")

        writer = csv.writer(f)
        for images in get_from_photos():
            image_urls = [i[0] for i in images if i[1] not in existing_ids]

            skipped = len(images) - len(image_urls)
            to_do = len(image_urls)

            if skipped > 0:
                total_skipped += skipped
                print(f"\rSkipping {skipped}, total skipped: {total_skipped/existing_count*100 // 1}%", end="")

            if to_do == 0:
                continue

            texts = image_to_text(image_urls)

            for i, text in enumerate(texts):
                writer.writerow([images[i][1], text])

            total_processed += to_do

            print(f"\rProcessed {to_do}, This run {total_processed}, Total {total_processed + existing_count}", end="")

    print(f"\n\nTotal skipped: {total_skipped}")
    print(f"Total processed: {total_processed}")


if __name__ == '__main__':
    main()
