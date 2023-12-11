from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    employee_name = request.form.get('employee_name')

    image_folder = os.path.join('data/raw_images', employee_name)
    os.makedirs(image_folder, exist_ok=True)

    for i, image in enumerate(request.files.getlist('images')):

        image_path = os.path.join(image_folder, f'{employee_name}_{i}.jpg')
        image.save(image_path)

    
    return jsonify({'message':'Images recieved'})
if __name__ == "__main__":
    app.run(debug=True, host='192.168.2.128', port=5000)