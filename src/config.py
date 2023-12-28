import json

INPUT_PATH = "data/raw_images" #input of the raw images
OUTPUT_PATH = "data/yolo_annotations" #output of the yolo annotations
MODEL_PATH = "model" #all information to the existing model
ROOT_PATH = "./app"

def getLatestModelName() -> str:

    with open("model/version.json") as file:
            # in this json file the information about the latest model is stored
            data = json.load(file)
            return data["name"]

def getLatestModelVersion():

    with open("model/version.json") as file:
            # in this json file the information about the latest model is stored
            data = json.load(file)
            return int(data["version"])