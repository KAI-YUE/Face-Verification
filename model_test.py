import os
import cv2
import torch
import numpy as np

from src.model import Net
from src.dataset import FaceDataset
from src.rectify import Rectifier

if __name__ == "__main__":
    
    data_dir = r'D:\YUE\Courses\ML\LFW'
    model_dir = r'D:\Projects\Face-Verification\checkpoints\epoch1_.pth'

    threshold = 0.5
    load_from_scratch = False
    R = Rectifier() 
    fData = FaceDataset()

    # Load data first
    if load_from_scratch:
        for root, dirs, files in os.walk(data_dir):
            if files != []:
                pair = []
                for f in files:
                    img = cv2.imread(os.path.join(root, f))  
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Rectify
                    theta = R.estimate_rot(gray)
                    dst = R.rectify(gray, theta)

                    # Normalize
                    dst = fData.sqi_norm(dst)

                    min_value = np.min(dst)
                    max_value = np.max(dst)

                    pair.append((dst.flatten() - min_value)/(max_value - min_value))

                fData.X.append(np.hstack(pair))   # Normalize the data to [0, 1]
                                
                if "mismatch" in root:
                    fData.Y.append(0)
                else:
                    fData.Y.append(1)
        
        fData.X = np.asarray(fData.X)
        fData.Y = np.asarray(fData.Y)
        fData.save('test_data.json')
    
    else:
        fData.load('test_data.json')

    print("Load completed.")

    model_dict = torch.load(model_dir)
    model = Net()

    device = torch.device("cuda")
    model.load_state_dict(model_dict['nn_state_dict'])
    model = model.to(device)

    confusion_matrix = np.zeros((2, 2))

    for i in range(fData.Y.shape[0]):
        sample = torch.zeros((2, 1, fData.face_size, fData.face_size))
        sample[0,0,:,:] = torch.from_numpy(fData.X[i, :fData.face_area].reshape((fData.face_size, fData.face_size)))
        sample[1,0,:,:] = torch.from_numpy(fData.X[i, fData.face_area:].reshape((fData.face_size, fData.face_size)))

        pred = model(sample.to(device)).cpu().detach().numpy()
        similarity = pred[0] @ pred[1] 
        norm1 = np.linalg.norm(pred[0])
        norm2 = np.linalg.norm(pred[1])
        if not (norm1 < 1e-8 or norm2 < 1e-8):
            similarity /= norm1*norm2

        if similarity > threshold:
            if fData.Y[i] == 1:
                confusion_matrix[0, 0] += 1
            else:
                confusion_matrix[0, 1] += 1
        else:
            if fData.Y[i] == 1:
                confusion_matrix[1, 0] += 1
            else:
                confusion_matrix[1, 1] += 1

        # print(similarity)

    
    print(confusion_matrix)
    print("ACC: {}".format((confusion_matrix[0,0] + confusion_matrix[1, 1])/np.sum(confusion_matrix)))
