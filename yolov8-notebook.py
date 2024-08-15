
# setup gpu to be used
# -1 means: disable cuda/gpu only using cpu
# otherwise the number will indicate gpu id
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
class YOLOv8:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')

        self.train_data = './data/traffic_car/training_images'
        self.train_csv = './data/traffic_car/train_solution_bounding_boxes.csv'
        self.test_data = './data/traffic_car/testing_images'

        self.root_dir = './YOLO/data_set'
        self.labels_dir = './labels'
        self.images_dir = './images'

        self.width = 676
        self.height = 380

        # Create directories if they don't exist
        os.makedirs(self.labels_dir + '/train', exist_ok=True)
        os.makedirs(self.labels_dir + '/val', exist_ok=True)
        os.makedirs(self.images_dir + '/train', exist_ok=True)
        os.makedirs(self.images_dir + '/val', exist_ok=True)

        self._prepare_data()

    def _prepare_data(self):
        df = pd.read_csv(self.train_csv)
        df['class'] = 0
        df.rename(columns={'image': 'image_name'}, inplace=True)

        # Calculate the center coordinates, width, and height for YOLO format
        df['x_centre'] = (df['xmin'] + df['xmax']) / 2
        df['y_centre'] = (df['ymin'] + df['ymax']) / 2
        df['width'] = (df['xmax'] - df['xmin'])
        df['height'] = (df['ymax'] - df['ymin'])

        # Normalize bounding box coordinates
        df['x_centre'] = df['x_centre'] / self.width
        df['y_centre'] = df['y_centre'] / self.height
        df['width'] = df['width'] / self.width
        df['height'] = df['height'] / self.height

        self.df_yolo = df[['image_name', 'class', 'x_centre', 'y_centre', 'width', 'height']]

        # Save labels in YOLO format
        self.save_labels()

    def save_labels(self):
        img_list = list(sorted(os.listdir(self.train_data)))
        np.random.shuffle(img_list)

        for i, img_name in enumerate(img_list):
            subset = 'train'
            if i >= 0.8 * len(img_list):  # Use 80% of data for training, 20% for validation
                subset = 'val'

            if np.isin(img_name, self.df_yolo['image_name'].values):
                columns = ['class', 'x_centre', 'y_centre', 'width', 'height']
                img_box = self.df_yolo[self.df_yolo['image_name'] == img_name][columns].values
                label_path = os.path.join(self.labels_dir, subset, img_name[:-4] + '.txt')

                with open(label_path, 'w+') as f:
                    for row in img_box:
                        text = " ".join(map(str, row))
                        f.write(text)
                        f.write('\n')

            old_image_path = os.path.join(self.train_data, img_name)
            new_image_path = os.path.join(self.images_dir, subset, img_name)
            shutil.copy(old_image_path, new_image_path)

    def train_model(self, epochs=10, batch_size=16):
        # Save YAML file for training configuration
        yolo_format = {
            'path': self.root_dir,
            'train': os.path.join(self.images_dir, 'train'),
            'val': os.path.join(self.images_dir, 'val'),
            'nc': 1,
            'names': {0: 'car'}
        }

        with open(os.path.join(self.root_dir, 'yolo.yaml'), 'w') as outfile:
            yaml.dump(yolo_format, outfile, default_flow_style=False)

        # Train the model
        self.model.train(data=os.path.join(self.root_dir, 'yolo.yaml'), epochs=epochs, patience=5, batch=batch_size, lr0=0.001, imgsz=640)

    def predict(self, conf=0.5, iou=0.75):
        # Load the best model weights
        path_best_weights = os.path.join(self.root_dir, 'runs/detect/train/weights/best.pt')
        self.model = YOLO(path_best_weights)

        # Make predictions on the test data
        results = self.model.predict(source=self.test_data, conf=conf, iou=iou)

        prediction_dir = os.path.join(self.root_dir, 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)

        test_img_list = []
        for result in results:
            if len(result.boxes.xyxy):
                name = result.path.split("/")[-1].split(".")[0]
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

    def visualize_results(self, test_img_list):
        def show_bbox(img, boxes, scores, axis, color=(0, 255, 0)):
            boxes = boxes.astype(int)
            img = img.copy()
            for i, box in enumerate(boxes):
                score = f"{scores[i]:.4f}"
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 10
                cv2.putText(img, score, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            axis.imshow(img)
            axis.axis("off")

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        ax = axes.flatten()

        imgs_name = np.random.choice(test_img_list, 4)

        for i, img_name in enumerate(imgs_name):
            img_file_path = os.path.join(self.test_data, img_name + ".jpg")
            img = cv2.imread(img_file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            label_file_path = os.path.join(self.root_dir, 'predictions', img_name + ".txt")
            label = pd.read_csv(label_file_path, sep=" ", header=None).values
            scores = label[:, 0]
            boxes = label[:, 1:]
            show_bbox(img, boxes, scores, axis=ax[i])

        plt.savefig("car.png")

if __name__ == "__main__":
    yolo = YOLOv8()
    yolo.train_model(epochs=10)
    test_img_list = yolo.predict()
    yolo.visualize_results(test_img_list)



path_best_weights="D:\Python/sup/YOLO/runs/detect/train8/weights/best.pt"
model = YOLO(path_best_weights) 

metrics = model.val() 


# In[ ]:


print(f'mean average precision @ .50: {metrics.box.map50}')


# In[ ]:


with torch.no_grad():
    results = model.predict(source = test_data, conf=0.5, iou=0.75)


# In[ ]:


# get_ipython().system("mkdir -p = './data_set/predictions'")
prediction_dir = './data_set/predictions'


# In[ ]:


test_img_list=[]
for result in results:
    if len(result.boxes.xyxy):
        name=result.path.split("/")[-1].split(".")[0]
        boxes=result.boxes.xyxy.cpu().numpy()
        scores=result.boxes.conf.cpu().numpy()
        
        test_img_list.append(name)
        
        label_file_path=os.path.join(prediction_dir,name+".txt")
        with open(label_file_path,"w+") as f:
            for score,box in zip(scores,boxes):
                text=f"{score:0.4f} "+" ".join(box.astype(str))
                f.write(text)
                f.write("\n")


# In[ ]:


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


# In[ ]:


fig,axes=plt.subplots(2,2,figsize=(12,12))
plt.subplots_adjust(wspace=0.1,hspace=0.1)
ax=axes.flatten()

imgs_name=np.random.choice(test_img_list,4)

for i,img_name in enumerate(imgs_name):
    img_file_path=os.path.join(test_data,img_name+".jpg")
    img=cv2.imread(img_file_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    label_file_path=os.path.join(prediction_dir,img_name+".txt")
    label=pd.read_csv(label_file_path,sep=" ",header=None).values
    scores=label[:,0]
    boxes=label[:,1:]
    show_bbox(img,boxes,scores,axis=ax[i])
    
plt.savefig("car.png")


# In[ ]:





# In[ ]:




