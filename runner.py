import subprocess

# Chạy chức năng predict của TestImage
subprocess.run("python main.py test_image /path/to/vid_5_26560.jpg predict 0.5 0.75", shell=True)

# Chạy chức năng show_class của TestImage
subprocess.run("python main.py test_image /path/to/vid_5_26560.jpg show_class", shell=True)

# Chạy chức năng show_model của TestImage
subprocess.run("python main.py test_image /path/to/vid_5_26560.jpg show_model", shell=True)

# Chạy chức năng show của TestImage
subprocess.run("python main.py test_image /path/to/vid_5_26560.jpg show", shell=True)

# Chạy chức năng prepare_data của YOLOv8
subprocess.run("python main.py yolo prepare_data", shell=True)

# Chạy chức năng train_model của YOLOv8
subprocess.run("python main.py yolo train_model", shell=True)

# Chạy chức năng predict của YOLOv8
subprocess.run("python main.py yolo predict", shell=True)

# Chạy chức năng draw_bounding_boxes của YOLOv8
subprocess.run("python main.py yolo draw_bounding_boxes ./data/testing_images/vid_5_420.jpg ./data_set/predictions/vid_5_420.txt", shell=True)
