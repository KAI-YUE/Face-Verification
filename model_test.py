import os
import cv2
import time
import numpy as np

from model.net import Net
from src.rectify import Rectifier

if __name__ == "__main__":
    
    data_dir = r'D:\YUE\Courses\ML\LFW_toy'
    model_dir = r'.\checkpoints\model.dat'
    outfile=open('D:\\outputfile.txt','w')
    
    threshold = 83.337
    R = Rectifier() 
    NN = Net()
    NN.load(model_dir)

    input_size = 250
    input_area = input_size**2
    
    start = time.time()
    dist = []
    for root, dirs, files in os.walk(data_dir):
        if files != []:
            pair = []
            sample = np.zeros((2,input_size,input_size,3))
            for f in files:
                img = cv2.imread(os.path.join(root, f))  
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Rectify
                theta = R.estimate_rot(gray)
                dst = R.rectify_rgb(img, theta).astype(np.float)
                
                dst = cv2.resize(dst, (250, 250))
                dst -= np.array([103.939, 116.779, 123.68]).reshape(1,1,3)
                pair.append(dst)
                
            img1 = pair[0]
            img2 = pair[1]
            sample[0,:,:,:] = img1.reshape((1,input_size,input_size,3))
            sample[1,:,:,:] = img2.reshape((1,input_size,input_size,3))
            pred = NN.netforward(sample)
            
            dist.append(np.sqrt(np.sum((pred[0]-pred[1])**2)))
            
            if dist[-1]<threshold:
                pred_label = 1
                outfile.write('1\n')
            else:
                pred_label = 0
                outfile.write('0\n')
            
            print("distance:{:.3f}, pred:{}".format(dist[-1], pred_label))
                  
    print("{:.3f} h has elapsed.".format((time.time()-start)/3600))
    outfile.close()
