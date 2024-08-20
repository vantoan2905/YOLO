# gpu version for training
# -1 means: disable cuda/gpu only using cpu
# otherwise the number will indicate gpu id
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# libary 
import pandas as pd
import cv2
import shutil
import yaml
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch
import sys



# model using yolov8
class YOLOv8:
    def __init__(self):
        # Load model
        self.model = YOLO('yolov8s.pt')
        # Data path 
        self.train_data = './data/training_images'
        self.test_data = './data/testing_images'
        # size of image
        self.width = 676
        self.height = 380

        self.train_csv = './data/train_solution_bounding_boxes.csv'
        self.root_dir = './data_set'
        self.labels_dir = './data_set/labels'
        self.images_dir = './data_set/images'

        # # Create directories if they don't exist
        # os.makedirs(self.labels_dir + './data_set/train', exist_ok=True)
        # os.makedirs(self.labels_dir + './data_set/val', exist_ok=True)
        # os.makedirs(self.images_dir + './data_set/train', exist_ok=True)
        # os.makedirs(self.images_dir + './data_set/val', exist_ok=True)


    def _prepare_data(self):
        df = pd.read_csv(self.train_csv)
        df['class'] = 0
        # Rename the 'image' column to 'image_name'
        df.rename(columns={'image': 'image_name'}, inplace=True)
        df["x_centre"] = (df["xmin"] + df["xmax"]) / 2
        df["y_centre"] = (df["ymin"] + df["ymax"]) / 2
        df["width"] = (df["xmax"] - df["xmin"])
        df["height"] = (df["ymax"] - df["ymin"])
        # Normalize bounding box coordinates
        df["x_centre"] = df["x_centre"] / self.width
        df["y_centre"] = df["y_centre"] / self.height
        df["width"] = df["width"] / self.width
        df["height"] = df["height"] / self.height
        # Save labels in YOLO format
        self.df_yolo = df[["image_name", "class", "x_centre", "y_centre", "width", "height"]]
        imag_list = list(sorted(os.listdir(self.train_data)))
        np.random.shuffle(imag_list)
        for i, image_name in enumerate(imag_list):
            subset = 'train'
            if i >= 0.8 * len(imag_list):  
                subset = 'val'
            if np.isin(image_name, self.df_yolo["image_name"]):
                self.columns = ['class', 'x_centre', 'y_centre', 'width', 'height']
                img_box = df[df["image_name"] == image_name][self.columns].values
                label_path = os.path.join(self.labels_dir, subset, image_name[:-4] + '.txt')
                with open(label_path, 'w+') as f:
                    for row in img_box:
                        text = " ".join(row.astype(str))
                        f.write(text)
                        f.write('\n')
            
            old_image_path = os.path.join(self.train_data, image_name)
            new_image_path = os.path.join(self.images_dir, subset, image_name)
            try:
                shutil.copy(old_image_path, new_image_path)
            except FileNotFoundError:
                print("Error while copying image: {} to {}""".format(old_image_path, new_image_path))
        
        yolo_format = dict(path = './data_set',
                  train='./data_set/images/train',
                  val ='./data_set/images/val',
                  nc=1,
                  names={0:'car'})
        with open('./data_set/yolo.yaml', 'w') as outfile:
            yaml.dump(yolo_format, outfile, default_flow_style=False)
        print("Data prepared successfully!")

    # Train the model method
    def train_model(self, epochs=50, batch_size=16):
        self.model.train(
            data='./data_set/yolo.yaml',
            patience = 5,
            epochs=epochs,
            batch=batch_size,
            workers=3
        )
    def val(self):
        path_best_weights="./runs/detect/train24/weights/best.pt"
        model = YOLO(path_best_weights) 
        metrics = model.val() 
        print(f'mean average precision @ .50: {metrics.box.map50}')

    def predict(self):
        prediction_dir = './data_set/predictions'
        with torch.no_grad():
            results = self.model.predict(source = path_to_data_test , conf=0.5, iou=0.75)
        test_img_list = []
        for result in results:
            if len(result.boxes.xyxy) :
                name = result.path.split("\\")[-1].split(".")[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                test_img_list.append(name)
                label_file_path = os.path.join(prediction_dir, name + ".txt")
                with open(label_file_path, "w+") as f:
                    for score, box in zip(scores, boxes):
                        text = f"{score:0.4f} " + " ".join(map(str, box))
                        f.write(text)
                        f.write("\n")
        return test_img_list


    def draw_bounding_boxes(self, image_path, label_path):
        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            _, x_min, y_min, x_max, y_max = map(float, line.strip().split())
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()






if __name__ == '__main__':

    if sys.argv[1] == 'yolo':
        yolo = YOLOv8()
        if sys.argv[2] == 'prepare_data':
            yolo._prepare_data()
        elif sys.argv[2] == 'train_model':
            yolo.train_model()
        elif sys.argv[2] == 'predict':
            yolo.predict()
        elif sys.argv[2] == 'draw_bounding_boxes':
            yolo.draw_bounding_boxes(sys.argv[3], sys.argv[4])

