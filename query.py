import requests
import urllib.request
import argparse
from os import remove

# url = "http://localhost:8080/predict" # <--- Uncomment this line to test on localhost
url = "https://landscape-recognition-6h5ffr6b6a-uc.a.run.app/predict" # <--- Uncomment this line to test on cloud deploy
# ^ Make sure only one line is activated at a time to prevent conflicts ^ 

parser = argparse.ArgumentParser(description="Pass a URL to a .jpg image.")
parser.add_argument('image_url', type=str, help="input a url to a jpg image.")
args=parser.parse_args()

img_path = args.image_url

urllib.request.urlretrieve(img_path, 'image.jpg')

with open('image.jpg', 'rb') as img_file:
    payload = {"img": img_file}
    print(requests.post(url, files=payload).json())

remove('./image.jpg')