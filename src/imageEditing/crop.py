# Script that takes all images in the folder and splits them in half
import cv2 
import os
from os import listdir

folderDir = "../StereoImages/Total"
pathL = "../StereoImages/Left"
pathR = "../StereoImages/Right"
for imageName in os.listdir(folderDir):
    # do stuff 
    imagePath = os.path.join(folderDir, imageName)
    img = cv2.imread(imagePath)
    cv2.imshow("image",img)

    h, w, channels = img.shape

    left_part = img[:, :w//2] 
    right_part = img[:, w//2:]  
    
    cv2.imwrite(os.path.join(pathR , f'right_{imageName}'), right_part)
    cv2.imwrite(os.path.join(pathL , f'left_{imageName}'), left_part)





