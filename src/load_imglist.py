# Python Libraries
import cv2
import os
import json
import logging
import numpy as np

# PyTorch Libraries
import torch
import torch.utils.data as data


class ImageList(data.Dataset):
    def __init__(self, **kwargs):
        
        self.face_size = 100
        self.face_area = self.face_size**2

        self.X = None
        self.Y = None

        if "src_dir" in kwargs:
            self.load(kwargs["src_dir"])
            self.save_mat("samples.json")
            print("Total samples:{}".format(self.X.shape[0]))
        else:
            self.load_matrix("samples.json")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.ERROR)
        fh = logging.FileHandler('log.txt', 'w')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

    ####################### Public Methods ########################
    
    def set_param(self, **kwargs):
        """
        Set parameters of the model
        """
        for arg, value in kwargs.items():
            command = "self.{}".format(arg)
            exec(command + "={}".format(value))
    
    def load(self, src_dir, key_word=""):
        """
        Load the all of the faces to the matrix.
        -----------------------------------------------------------
        Args:
            src_dir:    root directory of the dataset.
            key_word:   key word in the file name. 'rectified', 'funneled', ect.
        -----------------------------------------------------------
        Returns:
            dict containing X, Y
            X:          data matrix. Each row is the flattened pixels of a pair.
            Y:          labels corresponding to different people.
        """
        self.X = []
        self.Y = []
        label = 0
        for root, dirs, files in os.walk(src_dir):
            if files != []:
                for file in files:
                    if key_word in file:
                        img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                        min_value = np.min(img)
                        max_value = np.max(img)
                        self.X.append((img.flatten() - min_value)/(max_value - min_value))   # Normalize the data to [0, 1]
                        self.Y.append(label)
                label +=1
        
        self.X = np.asarray(self.X)
        self.Y = np.asarray(self.Y)

    def save_mat(self, dst_dir):
        with open(dst_dir, "w") as fp:
            json.dump(dict(X = self.X.tolist(), Y = self.Y.tolist()), fp, indent=4)
    
    def load_mat(self, src_dir):
        with open(src_dir, "r") as fp:
            a_set = json.load(fp)

        self.X = np.asarray(a_set["X"])
        self.Y = np.asarray(a_set["Y"])

    ######################## Private Methods ##########################
    def __getitem__(self, idx):
        data = torch.from_numpy(self.X[idx, :self.face_area].reshape((self.face_size, self.face_size)))
        data = data.to(torch.float)
        
        label = torch.tensor(self.Y[idx])
        label = label.to(torch.long)
        
        data = data.view(-1, data.shape[-2], data.shape[-1])

        return dict(image=data, label=label)

    def __len__(self):
        return self.Y.shape[0]