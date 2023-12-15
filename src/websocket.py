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
    if (isTraining):
        jsonify({"message":"There is currently a model trained. Please try again later"}), 501
        print("Message while training, error 501 sent")
    else:
        semaphore.acquire()
        isTraining = True

        employee_name = request.form.get('employee_name')

        image_folder = os.path.join('data/raw_images', employee_name)
        os.makedirs(image_folder, exist_ok=True)

        for i, image in enumerate(request.files.getlist('images')):

            image_path = os.path.join(image_folder, f'{employee_name}_{i}.jpg')
            image.save(image_path)

        file_count = len([file for file in os.listdir(image_folder) if file.lower().endswith('.jpg')])
        if (file_count >= 60):
            jsonify({"message":"Training has started. This might take some time."})
            try:
                train.detect_buttons_and_create_annotations(employee_name)
                train.create_config_yaml(employee_name)
                train.train_new_model(employee_name)

                print("Training done")
                jsonify({"message":"Training sucessful, new model can be downloaded"})
            except Exception as e:
                jsonify({"message":f"An error has occured. {e}"}), 500
        else:
            jsonify({"message":f"There are {file_count} images. Minimum requirement is 60"}), 400
        
        isTraining = False

        semaphore.release()

@app.route('/update', methods=['GET'])
def update():

    user_version = request.form.get('version')

    try:
        with open(os.path.join(MODEL_PATH), 'r') as file:
            data = json.load(file)
            local_version = data["version"]

            if (user_version < local_version):
                latest_name = data["name"]

                torchscript_path = os.path.join(MODEL_PATH,latest_name,"weights/best.torchscript")
                send_file(torchscript_path, as_attachment=True, download_name=latest_name)

                jsonify({"message":"File successfully sent"})
            else:
                jsonify({"message":"Already up to date"})

    except FileNotFoundError:
        return jsonify({"message":"File not found"}), 500

    except Exception as e:
        return jsonify({"message":f"An error has occured while sending the file. {e}"}), 500

    



if __name__ == "__main__":
    app.run(debug=True, host='192.168.2.128', port=5000, threading=True)