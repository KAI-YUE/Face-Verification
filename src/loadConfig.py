"""
Read configurations from config.json.
"""
# Python Libraries
import json
import os

class DictClass(object):
    """
    Turns a dictionary into a class
    """
 
    def __init__(self, dictionary):
        """Constructor"""
        self.current_path = os.path.dirname(os.path.dirname(__file__))
        for key in dictionary:
            if (isinstance(dictionary[key], str)):
                dictionary[key] = os.path.join(self.current_path, dictionary[key])
                
                if not os.path.exists(dictionary[key]):
                    os.mkdir(dictionary[key]) 
            
            setattr(self, key, dictionary[key])
 
    def __repr__(self):
        """"""
        return "<DictClass: {}>".format(self.__dict__)
    
def loadConfig(file_name):
    with open(file_name, "r") as fp:
        config = json.load(fp)
    
    return DictClass(config) 
    
if __name__ == '__main__':
    test = loadConfig()
    