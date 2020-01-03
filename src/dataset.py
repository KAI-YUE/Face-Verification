# Python Libraries
import os
import cv2
import numpy as np
import logging
import re
import time 
import matplotlib.pyplot as plt

# My Libraries
from src.rectify import Rectifier

class FaceDataset(object):
    def __init__(self, **kwargs):
        self.face_size = 100
        self.face_area = self.face_size**2
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.ERROR)
        fh = logging.FileHandler('log.txt', 'w')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

    ###################### Public Methods #######################
    
    def load_pair_matrix(self, src_dir, key_word="rectified"):
        """
        Load the all of the face pairs to the matrix.
        -----------------------------------------------------------
        Args:
            src_dir:    root directory of the dataset.
            key_word:   key word in the file name. 'rectified', 'funneled', ect.
        -----------------------------------------------------------
        Returns:
            dict containing X, Y
            X:          data matrix. Each row is the flattened pixels of a pair.
            Y:          labels to indicate if the pair is matched or not.
        """
        X = []
        Y = []
        for root, dirs, files in os.walk(src_dir):
            if files != []:
                pair = []
                for file in files:
                    if key_word in file:
                        img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                        min_value = np.min(img)
                        max_value = np.max(img)
                        pair.append((img.flatten() - min_value)/(max_value - min_value))    # Normalize the data to [0, 1]
                
                X.append(np.hstack(pair))
                if "mismatch" in root:
                    Y.append(0)
                else:
                    Y.append(1)
        
        return dict(X = np.asarray(X), 
                    Y = np.asarray(Y))
                
    
    def disp_pair(self, X, Y, index=0):
        """
        Display a pair in the pair matrix.
        """
        fig = plt.figure()
        axs = fig.gca()
        axs.set_axis_off()
        axs.set_title("Matched {}".format(Y[index]))
        
        ax1, ax2 = fig.subplots(1, 2)
        ax1.imshow(X[index, :self.face_area].reshape((self.face_size, self.face_size)),
                   cmap='gray')
        ax1.set_axis_off()
        ax2.imshow(X[index, self.face_area:].reshape((self.face_size, self.face_size)),
                   cmap='gray')
        ax2.set_axis_off()
        
        plt.show()
        
    def load_matrix(self, src_dir, key_word="funneled"):
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
        X = []
        Y = []
        label = 0
        for root, dirs, files in os.walk(src_dir):
            if files != []:
                for file in files:
                    if key_word in file:
                        img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                        min_value = np.min(img)
                        max_value = np.max(img)
                        X.append((img.flatten() - min_value)/(max_value - min_value))   # Normalize the data to [0, 1]
                        Y.append(label)
                label +=1
        
        return dict(X = np.asarray(X), 
                    Y = np.asarray(Y))
        
    def Transform(self, src_dir, dst_dir, funneled_dir=None):
        """
        Transform the original dataset to the normalized cropped dataset.
        ------------------------------------------------------------
        Args:
            src_dir:      root directory of the original dataset.
            dst_dir:      root directory of the destination dataset.
            funneled_dir: root directory of the standard funneled dataset.
                          Not necessary. For comparison purpose.
        ------------------------------------------------------------
        """
        R = Rectifier()

        # i = 0

        start = time.time()
        for root, dirs, files in os.walk(src_dir):
            
            new_path = root.replace(src_dir, dst_dir)
            
            if (not os.path.exists(new_path)):
                os.mkdir(new_path)
            
            if files != []:
        #        index = re.search(r"\d\d\d\d", root)
        #        if not index is None:
        #            j = int(root[index.start():index.end()])
        #            if j >= 783:
        #                break
                    
                for file in files:
                    img = cv2.imread(os.path.join(root, file))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    theta = R.estimate_rot(gray)
                    img = R.rectify(gray, theta)
                    
                    index = re.search('_\d\d\d\d.jpg', file)
                    self.logger.info(root + file + " : {}".format(theta))
                    cv2.imwrite(os.path.join(new_path, file[:index.start()+5] + "_rectified.jpg"), img)
                    
                    if not funneled_dir is None:
                        name = file[:index.start()]
                        new_file_name =  file[:index.start()+5] + "_funneled.jpg"
                        funneled_img = cv2.imread(os.path.join(funneled_dir, name, file))
                        funneled_img = cv2.cvtColor(funneled_img, cv2.COLOR_BGR2GRAY)
                        funneled_img = R.rectify(funneled_img)
                        cv2.imwrite(os.path.join(new_path, new_file_name), funneled_img)                    
                    
        #    i += 1
        #    if (i >= 20):
        #        break
            
        print("== {:.2f} min has elasped ==".format((time.time()-start)/60))
        
    @staticmethod
    def sqi_norm(img):
        img_out = np.zeros_like(img)

        num_scale, num_param = para_Guafilter.shape
        
        scale_used = 0
        for inx_scale in range(num_scale):
        
            scale_used = scale_used + 1
            
            # get smoothed version
            hsize = int(para_Guafilter[inx_scale, 0])
            sigma = para_Guafilter[inx_scale, 1]
            img_smo = cv2.GaussianBlur(img, (hsize, hsize), sigma)
            img_smo = np.where(img_smo>0, img_smo, 1)

            # get self-quotient image
            QI_cur = img / img_smo
            
            # nonlinear transform 2: sigmoid transform
            QI_cur = 1 / (1 + np.exp(-QI_cur))
            QI_cur = (QI_cur - np.min(QI_cur))/ (np.max(QI_cur) - np.min(QI_cur))
            QI_cur = 255.0 * QI_cur
            
            # cumulation
            img_out = img_out + QI_cur

        # get the final self-quotient image
        img_out = img_out / scale_used
        img_out = img_out.astype('uint8')

        return img_out

para_Guafilter = \
np.array([[3,  0.5],  #1:  3x3
           [5,  1.0],  #2:  5x5
           [7,  2.0],  #3:  7x7
           [9,  2.0],  #4:  9x9
           [11,  3.0],  #5:  11x11
           [13,  3.8], #6:  13x13
           [15,  4.2],  #7:  15x15
           [17,  4.8],  #8:  17x17
           [19,  5.0],  #9:  19x19
           [21,  6.0],  #10: 21x21
           [23,  8.0],  #11: 23x23
           [25,  9.0]])


if __name__ == '__main__':
    src_dir = r"F:\DataSets\LFW_all\toy"
    dataset = FaceDataset()
    a_set = dataset.load_matrix(src_dir)
    
        
        