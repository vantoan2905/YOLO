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


# Test data
path_to_data_test = 'D:/Python/sup/YOLO/data/testing_images'
path_to_data_train = 'D:/Python/sup/YOLO/data/training_images'



# D:/Python/sup/YOLO/data/training_images/vid_4_940.jpg
# D:/Python/sup/YOLO/data/testing_images/vid_5_26560.jpg
# show one image in training dataset
from PIL import Image

class TestImage:
    def __init__(self, path):
        self.path = path
        self.model = YOLO('yolov8s.pt')
        self.image = Image.open(path, mode ='r')
    # Predict using default model non train with custom data
    def predict(self, conf, iou):
        # Make predictions on the test data
        
        result = self.model.predict(source=self.path, conf=conf, iou=iou)
        # boxes 
        for result in result:
            boxes = result[0].boxes
        print("boxes: ", boxes)
    def show_class(self):
        class_name = self.model.names
        print("class_name: ", class_name)
    # show model 
    def show_model(self):
        model = self.model
        print("model: ", model)
        
    def show(self):
        plt.imshow(self.image)
        plt.axis('off')
        plt.show()



class YOLOv8:
    def __init__(self):
        # Load model
        self.model = YOLO('yolov8s.pt')
        # Data path 
        self.train_data = path_to_data_train
        self.test_data = path_to_data_test
        # size of image
        self.width = 676
        self.height = 380

        self.train_csv = 'D:/Python/sup/YOLO/data/train_solution_bounding_boxes.csv'
        self.root_dir = 'D:/Python/sup/YOLO/data_set'
        self.labels_dir = 'D:/Python/sup/YOLO/data_set/labels'
        self.images_dir = 'D:/Python/sup/YOLO/data_set/images'

        # # Create directories if they don't exist
        # os.makedirs(self.labels_dir + 'D:/Python/sup/YOLO/data_set/train', exist_ok=True)
        # os.makedirs(self.labels_dir + 'D:/Python/sup/YOLO/data_set/val', exist_ok=True)
        # os.makedirs(self.images_dir + 'D:/Python/sup/YOLO/data_set/train', exist_ok=True)
        # os.makedirs(self.images_dir + 'D:/Python/sup/YOLO/data_set/val', exist_ok=True)


    def _prepare_data(self):
        """
        Prepares the data for training and validation by performing the following steps:
        1. Reads the CSV file containing the annotated data.
        2. Renames the 'image' column to 'image_name'.
        3. Calculates the center coordinates, width, and height for YOLO format.
        4. Normalizes the bounding box coordinates.
        5. Saves the labels in YOLO format.
        6. Shuffles the list of image names.
        7. Divides the list into training and validation subsets.
        8. Saves the labels in YOLO format txt.
        9. Copies the images to the appropriate directories.
        """
        
        # Read the CSV file
        df = pd.read_csv(self.train_csv)
        
        # Add a column for class
        df['class'] = 0
        
        # Rename the 'image' column to 'image_name'
        df.rename(columns={'image': 'image_name'}, inplace=True)
        
        # Calculate the center coordinates, width, and height for YOLO format
        df["x_centre"] = (df["xmin"] + df["xmax"]) / 2
        df["y_centre"] = (df["ymin"] + df["ymax"]) / 2
        df["width"] = (df["xmax"] - df["xmin"])
        df["height"] = (df["ymax"] - df["ymin"])
        
        # Normalize bounding box coordinates
        df["x_centre"] = df["x_centre"] / self.width
        df["y_centre"] = df["y_centre"] / self.height
        df["width"] = df["width"] / self.width
        df["height"] = df["height"] / self.height
        
        # Save the annotated data in YOLO format
        self.df_yolo = df[["image_name", "class", "x_centre", "y_centre", "width", "height"]]
        
        # Get the list of image names
        imag_list = list(sorted(os.listdir(self.train_data)))
        
        # Shuffle the list of image names
        np.random.shuffle(imag_list)
        
        # Divide the list into training and validation subsets
        for i, image_name in enumerate(imag_list):
            subset = 'train'
            if i >= 0.8 * len(imag_list):  # Use 80% of data for training, 20% for validation
                subset = 'val'
            
            # Save labels in YOLO format txt
            if np.isin(image_name, self.df_yolo["image_name"]):
                self.columns = ['class', 'x_centre', 'y_centre', 'width', 'height']
                img_box = df[df["image_name"] == image_name][self.columns].values
                label_path = os.path.join(self.labels_dir, subset, image_name[:-4] + '.txt')
                try:
                    with open(label_path, 'w+') as f:
                        for row in img_box:
                            text = " ".join(row.astype(str))
                            f.write(text)
                            f.write('\n')
                except:
                    # print("Error while saving label file: ", label_path)
                    print("a")
            
            # Copy the images to the appropriate directories
            old_image_path = os.path.join(self.train_data, image_name)
            new_image_path = os.path.join(self.images_dir, subset, image_name)
            try:
                shutil.copy(old_image_path, new_image_path)
            except FileNotFoundError:
                print("Error while copying image: {} to {}""".format(old_image_path, new_image_path))
        
        yolo_format = dict(path = 'D:/Python/sup/YOLO/data_set',
                  train='D:/Python/sup/YOLO/data_set/images/train',
                  val ='D:/Python/sup/YOLO/data_set/images/val',
                  nc=1,
                  names={0:'car'})
        with open('D:/Python/sup/YOLO/data_set/yolo.yaml', 'w') as outfile:
            yaml.dump(yolo_format, outfile, default_flow_style=False)
        print("Data prepared successfully!")

    # Train the model 
    def train_model(self, epochs=50, batch_size=16):
        self.model.train(
            data='./data_set/yolo.yaml',
            patience = 5,
            epochs=epochs,
            batch=batch_size,
            workers=3
        )
    def val(self):

        path_best_weights="D:\Python/sup/YOLO/runs/detect/train24/weights/best.pt"
        model = YOLO(path_best_weights) 
        metrics = model.val() 
        print(f'mean average precision @ .50: {metrics.box.map50}')

    def predict(self):
        path_best_weights="D:\Python/sup/YOLO/runs/detect/train24/weights/best.pt"

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
                print(prediction_dir)
                print(name)
                label_file_path = os.path.join(prediction_dir, name + ".txt")
                print (label_file_path)
                with open(label_file_path, "w+") as f:
                    for score, box in zip(scores, boxes):
                        text = f"{score:0.4f} " + " ".join(map(str, box))
                        f.write(text)
                        f.write("\n")

        return test_img_list

    def show_bbox(img,boxes,scores,axis,color=(0,255,0)):
        boxes=boxes.astype(int)
        scores=scores
        img=img.copy()
        for i,box in enumerate(boxes):
            score=f"{scores[i]:.4f}"
            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color,2)
            y=box[1]-10 if box[1]-10>10 else box[1]+10
            cv2.putText(img,score,(box[0],y),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        axis.imshow(img)
        axis.axis("off")
    def show_class(self):
        fig,axes=plt.subplots(2,2,figsize=(12,12))
        plt.subplots_adjust(wspace=0.1,hspace=0.1)
        ax=axes.flatten()

        imgs_name=np.random.choice(self.predict(),4)
        prediction_dir = './data_set/predictions'
        for i,img_name in enumerate(imgs_name):
           img_file_path=os.path.join(path_to_data_test,img_name+".jpg")
           img=cv2.imread(img_file_path)
           img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

           label_file_path=os.path.join(prediction_dir,img_name+".txt")
           label=pd.read_csv(label_file_path,sep=" ",header=None).values
           scores=label[:,0]
           boxes=label[:,1:]
           self.show_bbox(img,boxes,scores,axis=ax[i])

        plt.savefig("car.png")

if __name__ == '__main__':
    # image = TestImage(path_to_data_train + '/vid_5_26560.jpg')
    # image.predict(conf=0.5, iou=0.75)
    # image.show_class()
    # image.show_model()
    # image.show()

    yolo = YOLOv8()
    # yolo._prepare_data()

    # yolo.train_model()

    # yolo.predict()
    yolo.show_class()

