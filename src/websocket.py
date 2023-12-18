from flask import Flask, request, jsonify, send_file
from paths import MODEL_PATH
import threading
import os
import train
import json

app = Flask(__name__)
semaphore = threading.Semaphore()

@app.route('/upload', methods=['POST'])
def upload():
    #there should be only one training at the time, so this section is blocked while training
    if (isTraining):
        return jsonify({"message":"There is currently a model trained. Please try again later"}), 501
    else:
        semaphore.acquire()
        isTraining = True

        #get employee name and create directory for image storage
        employee_name = request.form.get('employee_name')
        image_folder = os.path.join('data/raw_images', employee_name)
        os.makedirs(image_folder, exist_ok=True)

        #for each image in the attchment it is saved to the directory as a jpg file
        for i, image in enumerate(request.files.getlist('images')):

            image_path = os.path.join(image_folder, f'{employee_name}_{i}.jpg')
            image.save(image_path)

        #for a good quality of the trained model there need to be at least 60 images for the training
        file_count = len([file for file in os.listdir(image_folder) if file.lower().endswith('.jpg')])
        if (file_count >= 60):
            try:
                # locates the button in each image and creates the annotation in yolo format for that image
                train.detect_buttons_and_create_annotations(employee_name)
                # creates the config.yaml that provides the neccessary paths and classes for training
                train.create_config_yaml(employee_name)
                # initializes the training based on the current model
                train.train_new_model(employee_name)

                print("Training done")

                isTraining = False
                semaphore.release()
                return jsonify({"message":"Training sucessful, new model can be downloaded"})
            
            except Exception as e:

                isTraining = False
                semaphore.release()
                return jsonify({"message":f"An error has occured. {e}"}), 500
        else:
            isTraining = False
            semaphore.release()
            return jsonify({"message":f"There are {file_count} images. Minimum requirement is 60"}), 400


@app.route('/updateTorchscript', methods=['GET'])
def update():
    #this route is used to pull the updated and exported model in torchscript format
    try:
        #information for the latest version is stored in the version.json file and is used to retrieve
        jsonPath = os.path.join(MODEL_PATH, "version.json")
        with open(jsonPath) as file:
            data = json.load(file)
            latest_name = data["name"]
            torchscript_path = os.path.join("..\\",MODEL_PATH,latest_name,"weights\\best.torchscript")
            return send_file(torchscript_path, as_attachment=True)

    except Exception as e:
        print(e)
        return jsonify({"message":f"An error has occured while sending the file. {e}"}), 500

@app.route('/updateLabels', methods=['GET'])
def update():
    # this route is for sending the coressponding classes.txt for the labels of the new model
    try:
        
        label_path = os.path.join("..\\",MODEL_PATH,"existingClasses.txt")
        return send_file(label_path, as_attachment=True)

    except Exception as e:
        print(e)
        return jsonify({"message":f"An error has occured while sending the file. {e}"}), 500


@app.route('/checkVersion', methods=['GET'])  
def checkVersion():
    # this route compares the server version to the app version
    # If the app is outdated it will get a 200 status code and pull the new files
    try:

        user_version = int(request.args.get('version'))
        with open("model/version.json") as file:
            # in this json file the information about the latest model is stored
            data = json.load(file)
            local_version = int(data["version"])

            if (user_version < local_version):
                # client gets a 200 status code and the current verion and latest added employee name
                latest_name = data["name"]
                return jsonify({"version":local_version, "name":latest_name})

            else:
                # client is still up to date - no pull needed
                return jsonify({"message":"Already up to date"}), 201

    except Exception as e:
        print(e)
        return jsonify({"message":f"An error has occured while sending the file. {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host='192.168.2.128', port=5000)