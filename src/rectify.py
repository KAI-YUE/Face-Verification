# Python Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

# My Libraries
from src.loadConfig import loadConfig, DictClass

class Rectifier(object):
    def __init__(self, config=None):
        
        if config is None:
            config = DictClass(default_config)
            
        self.face_cascade = cv2.CascadeClassifier(config.haar_face_param_dir)
        self.eye_cascade = cv2.CascadeClassifier(config.haar_eyes_param_dir)
        self.angle_range = int(config.orientation_range)
        self.angle_threshold = int(config.angle_threshold)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.ERROR)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        self.logger.addHandler(sh)
    
    
    ####################### Public Methods ########################
    
    def set_param(self, **kwargs):
        """
        Set parameters of the model
        """
        for arg, value in kwargs.items():
            command = "self.{}".format(arg)
            exec(command + "={}".format(value))
    
    def estimate_rot(self, img, angle_step=2):
        """
        Estimate the orientation angle of the face
        ------------------------------------------
        Args:
            img:         the input gray-scale image.
            angle_step:  step to increase between iterations.
        ---------------------------------------------
        Return:
            angle:       the estimated rotation angle 
                         (-, clockwise; +, anticlockwise)
        """
        started = False
        iterCount = 0
        sumAngle = 0
        
        img = self._crop(img)
        
        angle = -self.angle_range
        while angle <= self.angle_range:
            rotatedImg = self._rot_degree(img, angle)
            face_region = self.VJ_DetectFace(rotatedImg)
            
            if (not face_region is None): 
                if not started:
                    sumAngle = angle
                    iterCount = 1
                    started = True
                    prevAngle = angle
                else:
                    if (angle - prevAngle == angle_step):
                        sumAngle += angle
                        iterCount += 1
                        prevAngle = angle
                    else:
                        if iterCount*angle_step <= 15:
                            started = False
            angle += angle_step
        
        self.logger.info(iterCount)
        if iterCount * angle_step >= self.angle_threshold:
            return sumAngle / iterCount
        else:
            return 0
        
    def rectify(self, img, degree=0):
        """
        Rectify the img containing the face.
        -----------------------------------
        Args:
            img,        the input gray-scale image.
            degree,     degree to rotate. 
        -----------------------------------    
        Return:
            the cropped rectified image.
        """
        if degree:
            img = self._rot_degree(img, degree)
        
        rectified_img = self._crop(img, scale_factor=1.05)
        rectified_img = cv2.resize(rectified_img, (100, 100))
        return rectified_img
        
    
    def VJ_DetectFace(self, img, scale=1.1, neighborhood=3):
        """
        Detect the face with VJ algorithom. 
        ---------------------------------------------------------
        Args:
            img:            the input image to be detected.
            scale:          scaling factor
            neighborhood:   how many neighbors each candidate rectangle 
                            should have to retain it. 
        ------------------------------------------------------------
        Return:
            [x, y, w, h]:   A tetrad. ROI = img[y:y+h, x:x+w]
            (If faces are not detected, return None)
        """
        faces = self.face_cascade.detectMultiScale(img, scale, neighborhood)
        
        while (faces is () and scale-0.03 > 1):
            scale -= 0.03               # Automatically adjust the scaling factor
            faces = self.face_cascade.detectMultiScale(img, scale, neighborhood)
        
        if(faces is ()):
            return None
        else:
            if (faces.shape[0] == 1):
                (x,y,w,h) = faces[0]
            # Else if more than one faces have been detected, take the mean
            else:
                valid_face = []
                max_w_index = np.argmax(faces[:, 2], 0)
                max_h_index = np.argmax(faces[:, 3], 0)
                max_w = faces[max_w_index, 2]
                max_h = faces[max_h_index, 2]
                
                for i in range(faces.shape[0]):
                    if faces[i, 2] >= 0.8*max_w and faces[i, 3] >= 0.8*max_h:
                        if np.abs(faces[i, 0] - faces[max_w_index, 0]) > 0.5*max_w:
                            continue
                        else:
                            valid_face.append(faces[i])

                valid_face = np.asarray(valid_face)
                (x,y,w,h) = np.mean(valid_face, 0)
                
            return np.array([x,y,w,h], dtype=int)
        
    def VJ_DetectEyes(self, img, scale=1.1, neighborhood=5):
        """
        Detect the face with VJ algorithom. 
        ---------------------------------------------------------
        Args:
            img:            the input image to be detected.
            scale:          scaling factor
            neighborhood:   how many neighbors each candidate rectangle 
                            should have to retain it. 
        ------------------------------------------------------------
        Return:
            2*[x, y, w, h]:  Tetrads of eyes. ROI = img[y:y+h, x:x+w]
            (If eyes are not detected, return None)
       """
            
        eyes = self.eye_cascade.detectMultiScale(img, scale, neighborhood)
        
        while (eyes is () and scale-0.03 > 1):
            scale -= 0.03               # Automatically adjust the scaling factor
            eyes = self.eye_cascade.detectMultiScale(img, scale, neighborhood)
        
        if(eyes is ()):
            return None
        else:
            if (eyes.shape[0] == 2):
                result = eyes
            # Else if more than one faces have been detected, take the largest one
            else:
                index1 = np.argmax(eyes[:, 2])
                eyes[index1, 2] = 0 
                index2 = np.argmax(eyes[:, 2])
                result = eyes[[index1, index2]]

            return np.asarray(result)
        
    ######################## Private Methods ##########################
    
    @staticmethod
    def _rot_degree(img, degree):
        rows, cols = img.shape
        center = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(center, degree, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst
    
    def _crop(self, img, scale_factor=1.2):
        """
        Crop the ROI containing the face 
        """
        face_region = self.VJ_DetectFace(img, 1.2)
        
        # Eye region is utilized to correct the face region
        if not (face_region is None):
            [x, y, w, h] = face_region
            
            w_ = int(face_region[2] * scale_factor)
            h_ = int(face_region[3] * scale_factor)         
            x_ = int(max(face_region[0] - (w_ - face_region[2])/2, 0))
            y_ = int(max(face_region[1] - (h_ - face_region[3])/2, 0))
            
            return img[y_:y_+h_, x_:x_+w_].copy()
        else:           
            return img
            
#        eyes = self.VJ_DetectEyes(img[y:y+h, x:x+w])
        
#        if not eyes is None:
#            center_of_eyes = 1/2*(eyes[0, 0] + eyes[1, 0]) + \
#                             1/4*(eyes[0, 2] + eyes[1, 2])
#            x_ += max(int(center_of_eyes - w/2), 0)
#            
        

default_config = \
{
    "orientation_range":    20,
    "angle_threshold":      15,
    
    "haar_face_param_dir":  "./docs/haarcascade_frontalface_default.xml",
    "haar_eyes_param_dir":  "./docs/haarcascade_eye.xml"
}

if __name__ == "__main__":
    config = loadConfig("..\\config.json")
    R = Rectifier(config)
    
    src = r"D:\YUE\Courses\ML\LFW\match pairs\0007\Abdullah_al-Attiyah_0003.jpg"
    img = cv2.imread(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    theta = R.estimate_rot(gray)
    
    dst = R.rectify(gray, theta)
    
    io.imshow(dst)
    
