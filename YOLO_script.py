from random import shuffle, seed as randseed, randint
import os
import sys
import yaml
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import WindowsPath #!!#

# ------------------------------------------------------------------------------------------------- #
def YAM(dataset_location, Time2test=False):
    fold1 = 'valid'
    fold2 = 'test'
    if Time2test:
        fold2 = 'valid'
        fold1 = 'test'

    # define dataset configuration file (.yaml)
    with open(f"{dataset_location}/data.yaml", 'r') as f:
        dataset_yaml = yaml.safe_load(f)
    dataset_yaml["train"] = "../train/images"
    dataset_yaml["val"] = f"../{fold1}/images"
    dataset_yaml["test"] = f"../{fold2}/images"
    dataset_yaml["names"] = ['MD','ME','PD','PE']
    dataset_yaml["nc"] = 4
    with open(f"{dataset_location}/data.yaml", 'w') as f:
        yaml.dump(dataset_yaml, f)
# -------------------------------------------------------------------------------------------------- #

#- CROSS - VALIDATION -
dataset_location = "C:/Users/Utente/Desktop/BATCH2"
Images_name_list = os.listdir(f"{dataset_location}/dataset")
file_path = WindowsPath(f"{dataset_location}/dataset")

train_results = np.zeros([10, 2, 4]) #initializations
valid_results = np.zeros([10, 2, 4])
test_results = np.zeros([10, 2, 4])

randseed(0)
shuffle(Images_name_list)
#get a random 10% for test
foldlen = len(Images_name_list)//11
imgs2test = Images_name_list[-foldlen-2:]
testFlag = True
for k in range(10): #10-fold validation
    imgs4valid = Images_name_list[foldlen*k:foldlen*(k+1)]
    for img in Images_name_list:

        if img in imgs2test :
            if testFlag==True:
                folder = 'test'
            else: continue #jump to next img
        elif img in imgs4valid:
            folder = 'valid'
        else:
            folder = 'train'

        dest_path = WindowsPath(f"{dataset_location}/{folder}/images")
        if not dest_path.exists(): dest_path.mkdir()
        os.system(f"copy {file_path/img} {dest_path/img}")

    testFlag = False


    # -OBJECT DETECTION-
    # hyperparameters in exam
    f = 0
    B = 20
    E = 40
    #closs = [0.5, 0.4, 0.3]
    #wd = [0.05, 0.05, 0.005]

    # define dataset configuration file (.yaml) for training
    YAM(dataset_location)
    model = YOLO(f'yolov8n.pt')
    model.info()

    Theta0Seed = randint(0,100)
    print(Theta0Seed)
    results= model.train(
                    data=dataset_location+"/data.yaml",
                    imgsz=(320,240),
                    seed=Theta0Seed,
                    name= f"train_f={f}_BE={B}{E}_k={k}_{Theta0Seed}",
    # -Optimazier's Hyperparameter-
                    #optimizer='auto',  #choose between SGD,Adam,RAdam,AdamW,NAdam,RMSprop
                    epochs=E,
                    #patience=100,	#Number of epochs to wait without improvement in validation metrics before early
                                    # stopping the training. Helps prevent overfitting by stopping training when
                                    # performance plateaus.
                    batch=B,	    #Batch size, with three modes: set as an integer (e.g., batch=16),
                                    # auto mode for 60% GPU memory utilization (batch=-1),
                                    # or auto mode with specified utilization fraction (batch=0.70).

                    #lr0=0.01,
                    #lrf=0.01,
                    #weight_decay=0.05, #L2 regularization term
                    #momentum=0.9,
                    # dropout=0.0,	#Dropout rate for regularization in classification tasks,
                                    # preventing overfitting by randomly omitting units during training.

    # -Loss Function's Hyperparameter-
                    #box=7,	        #Weight of the box loss component in the loss function, influencing how much
                                    # emphasis is placed on accurately predicting bounding box coordinates.

                    #cls=0.3, 	    #Weight of the classification loss in the total loss function, affecting the
                                    # importance of correct class prediction relative to other components.

                    #dfl=0.5	,   #Weight of the distribution focal loss, used in certain YOLO versions
                                    # for fine-grained classification.

                    #pose=10.0,	    #Weight of the pose loss in models trained for pose estimation, influencing the
                                    # emphasis on accurately predicting pose keypoints.

    # -Utilities-
                    #save=True,     #Enables saving of training checkpoints and final model weights.
                                    # Useful for resuming training or model deployment.

                    #val=True,	    #Enables validation during training, allowing for periodic
                                    # evaluation of model performance on a separate dataset.
                    #plots=False,	#Generates and saves plots of training and validation metrics,
                                    # as well as prediction examples, providing visual insights into
                                    # model performance and learning progression.

                    freeze=f,      #Freezes the first N layers of the model or specified layers by index,
                           #None    # reducing the number of trainable parameters. Useful for fine-tuning
                                    # or transfer learning.

    # -Data Augumentation settings-
     # auto_augment='randaugment',	#Automatically applies a predefined augmentation policy (randaugment, autoaugment,
                                     # augmix), optimizing for classification tasks by diversifying the visual features.
                    scale=0,        #Scales the image by a gain factor, simulating objects at different distances from the camera.
                    fliplr=0.0,     #Flips the image left to right with the specified probability,useful for learning
                        # 0.0 - 1.0  # symmetrical objects and increasing dataset diversity.
                    mosaic=0,       #Combines four training images into one, simulating different scene compositions and
                       #0.0 - 1.0    # object interactions. Highly effective for complex scene understanding.
                   # erasing=0.4,   #Randomly erases a portion of the image during classification training, encouraging
                       # 0.0 - 0.9   # the model to focus on less obvious features for recognition.
                   # crop_fraction=0.5	#Crops the classification image to a fraction of its size to emphasize central
                       #0.1 - 1.0   # features and adapt to object scales, reducing background distractions.
                    hsv_h=0.1,	    #Adjusts the hue of the image by a fraction of the color wheel, introducing color
                       #0.0 - 1.0    # variability. Helps the model generalize across different lighting conditions.
                    hsv_s=0.1,		#Alters the saturation of the image by a fraction, affecting the intensity of colors.
                       #0.0 - 1.0     # Useful for simulating different environmental conditions.
                    hsv_v=0,	    #Modifies the value (brightness) of the image by a fraction, helping the model to
                       #0.0 - 1.0	  # perform well under various lighting conditions.

    )
    # yolo output is a tensor that contains the bounding box coordinates, objectness score, and class probabilities.
    mydevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_results[k, 0, :] = results.box.ap50
    train_results[k, 1, :] = results.box.ap

    # validation
    results = model.val(data=dataset_location + "/data.yaml",
                        name=f"valid_f={f}_BE={B}{E}_k={k}_{Theta0Seed}",
                        imgsz=640, batch=16, conf=0.6, iou=0.6, device=mydevice)
    valid_results[k, 0, :] = results.box.ap50
    valid_results[k, 1, :] = results.box.ap

    # testing
    YAM(dataset_location, True)
    results = model.val(data=dataset_location + "/data.yaml",
                        name=f"test_f={f}_BE={B}{E}_k={k}_{Theta0Seed}",
                        imgsz=640, batch=16, conf=0.6, iou=0.6, device=mydevice)
    test_results[k, 0, :] = results.box.ap50
    test_results[k, 1, :] = results.box.ap

    # destroy train and valid folder
    for folder in ['train','valid']:
        dest_path = WindowsPath(f"{dataset_location}/{folder}/images")
        os.system(f"rd /s /q {dest_path}")

# print results
print('mean_train =',np.mean(train_results, axis=0))
print('std_train =',np.std(train_results, axis=0))
print('mean_valid =',np.mean(valid_results, axis=0))
print('std_valid =',np.std(valid_results, axis=0))
print('mean_test =',np.mean(test_results, axis=0))
print('std_test =',np.std(test_results, axis=0))

