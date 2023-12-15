from ultralytics import YOLO
import os
from shutil import copy
from paths import INPUT_PATH, OUTPUT_PATH, ROOT_PATH, MODEL_PATH
import json

def detect_buttons_and_create_annotations(employee_name):

    # Structure of the YOLO Annotation is:
    #
    # EMPLOYEE_NAME
    #   classes.txt
    #   labels
    #       image_1.txt
    #       ...
    #   images
    #       image_1.jpg
    #       ...
    
    # define paths to input and output
    # create output structure for annotation
    input_folder = os.path.join(INPUT_PATH,employee_name)
    existing_classes_file = os.path.join(MODEL_PATH, "existingClasses.txt")#todo: this might has to be canged to the specific path


    output_folder = os.path.join(OUTPUT_PATH,employee_name)
    image_folder = os.path.join(output_folder, "images")
    label_folder = os.path.join(output_folder, "labels")

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    #the model will be named after the last learned employee, so we can read the last class and get the model
    with open(existing_classes_file, 'rb') as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        model_name = f.readline().decode()

    model = YOLO(os.path.join(MODEL_PATH, "detect_buttons.pt"))


    # open the classes from the old model, get the number of existing classes and write the new class to it
    # by stiping and appending we make sure there will no be a empty line or classes are squished together
    with open(existing_classes_file, "r") as file:
        existing_classes = [line.strip() for line in file.readlines()]

    new_object_class = len(existing_classes)

    if employee_name in existing_classes:
        new_object_class -= 1

    if employee_name not in existing_classes:
        with open(existing_classes_file, "a") as file:
            if file.tell() == 0:
                file.writelines(employee_name)
            else:
                file.writelines("\n"+ employee_name)


    for image in os.listdir(input_folder):
        #tests have shown that runnin this squentially is faster that giving the complete array at once to the model
        if image.lower().endswith((".png",".jpg",".jpeg")):

            #create paths for output files
            image_path = os.path.join(input_folder,image)
            output_image_path = os.path.join(image_folder, image)
            output_label_path = os.path.join(label_folder,image.replace(os.path.splitext(image)[-1], ".txt"))

            results = model(image_path)

            with open(output_label_path, "w") as label_file:
                # calculates for each box in one picture the values for the annotation format and then writes them to the specific file_name for the image
                for result in results:
                    boxes = result.boxes.cpu().numpy()

                    xyxys = boxes.xyxyn #this returns the normalised values for the top left and bottom right point of the rectangle

                    for xyxy in xyxys:
                        x1, y1, x2, y2 = xyxy

                        center_x = (x1+x2) / 2.0
                        center_y = (y1+y2) / 2.0

                        width = x2-x1
                        height = y2-y1

                        label_file.write("{} {} {} {} {} \n".format(new_object_class, center_x, center_y, width, height)) # class-name box-center-x box-center-y box-width box-height

            copy(image_path, output_image_path)
            copy(existing_classes_file, output_folder) #the class file is needed for training
                    
def create_config_yaml(employee_name):

    config_file_path = os.path.join(MODEL_PATH, "config.yaml")
    class_file_path = os.path.join(MODEL_PATH, "existingClasses.txt")

    dataset_root_dir = os.path.join(ROOT_PATH,OUTPUT_PATH, employee_name)

    config_content = f"path: {dataset_root_dir}\ntrain: images\nval: images\n\nnames:\n"


    with open(class_file_path, "r") as classes:
        existing_classes = [line.strip() for line in classes.readlines()]
        for index, class_name in enumerate(existing_classes):
            config_content += f"  {index}: {class_name}\n" 


    with open(config_file_path, 'w') as config_file:
        print(config_file_path)
        print("Write content to file:")
        print(config_content)
        config_file.write(config_content)

    copy(config_file_path, dataset_root_dir)

def train_new_model(employee_name):
    
    epochs = 1 #the epochs the model will be trained for
    existing_classes_file = os.path.join(MODEL_PATH, "existingClasses.txt")#todo: this might has to be canged to the specific path
    config_file = os.path.join(MODEL_PATH,"config.yaml")
    output_dir = os.path.join(MODEL_PATH, employee_name)

    #the model will be named after the last learned employee, so we can read the last class and get the model
    with open(existing_classes_file, 'r') as f:
        lines = f.readlines()

    if len(lines) >= 2:
        model_name = lines[-2].strip()
        last_model_path = os.path.join(MODEL_PATH, model_name,"weights\\best.pt")
    else:
        last_model_path = "yolov8n.pt" #if there is no previous model this is the default

    

    model = YOLO(last_model_path)

    model.train(data=config_file, epochs=epochs,name=output_dir, save=False, project=ROOT_PATH)

    model.export(format="torchscript",optimize=True)

    #update the json of the model
    with open(os.path.join(MODEL_PATH,"version.json"), 'rw') as file:
        data = json.load(file)
        current_verion = data['version']
        new_version = current_verion+1
        new_json_data = {'name': employee_name, 'version': new_version}
        json.dump(new_json_data)





    
if __name__ == "__main__":
    employee_name = "Claus Gustav"

    detect_buttons_and_create_annotations(employee_name)
    create_config_yaml(employee_name)
    train_new_model(employee_name)


