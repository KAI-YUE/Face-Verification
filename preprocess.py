# Python Libraries
import os 
import cv2
import re
import time

# My Libraries
from src.dataset import FaceDataset
from src.rectify import Rectifier

root_dir = r'D:\Downloads\Netdisk\test'
output_dir = r'D:\Downloads\Netdisk\test2'

R = Rectifier() 
fData = FaceDataset()

start = time.time()
counter = 0
for root, dirs, files in os.walk(root_dir):

    new_path = root.replace(root_dir, output_dir)
    if (not os.path.exists(new_path)):
        os.mkdir(new_path)

    if files != []:
        for f in files:       
            img = cv2.imread(os.path.join(root, f))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Rectify
            theta = R.estimate_rot(gray)
            dst = R.rectify(gray, theta)

            # Normalize
            dst = fData.sqi_norm(dst)
            cv2.imwrite(os.path.join(new_path, f), dst)
    
    counter += 1
    print("---------- {} finished, {:.3f} h has elapsed ---------------.".format(counter, (time.time()-start)/3600))