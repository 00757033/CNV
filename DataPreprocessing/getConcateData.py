import cv2
import os
import numpy as np
#合併多層OCTA、OCT影像
class Concatenation() :
    def __init__(self, PATH, data_class, layers = ["1_OCT", "2_OCT", "3_OCT", "4_OCT", "1", "2", "3", "4"]):
        self.PATH = PATH
        self.layers = layers
        self.data_class = data_class
        print("Start get concate data : " + data_class)
    #concateLayer需要合併的影像集
    def getConcate(self, path, concateLayer, ratio):
        setFloder(path + '/concate')
        for data in os.listdir(path+'/label'):
            images = []
            for i, layer in enumerate(concateLayer):
                image   = cv2.imread(path + '/' + layer + '/' + data)
                image   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image   = cv2.resize(image, (304,304))
                image   = np.expand_dims(image, axis = 2)
                if i > 0 :
                    image = ratio*image
                images.append(image)
            image_concate = np.append(images[0], images[1], axis=2)
            image_concate = np.append(image_concate, images[2], axis=2)
            cv2.imwrite(path + "/concate/" + data, image_concate)
    
    def getOrigin(self, path, layer):
        setFloder(path + '/origin')
        for data in os.listdir(path+'/label'):
            image   = cv2.imread(path + '/' + layer + '/' + data)
            image   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image   = cv2.resize(image, (304,304))
            cv2.imwrite(path + "/origin/" + data, image)

def createFloder(path_florder, layers):
    setFloder(path_florder)
    for layer in layers:
        setFloder(path_florder + '/' + layer)

def setFloder(path):
    if not os.path.isdir(path) : os.mkdir(path)

#使用範例如下    

def run(PATH_BASE):
    data_class  = 'PED'
    data_date   = '0106'
    data_groups  = ["OR", "CC"]
    dict_concate = {'OR': ["3_OCT", "4_OCT", "3"] , 'CC': ["4_OCT", "3_OCT", "4"]}
    dict_origin = {'OR': "3_OCT" , 'CC': "4_OCT"}
    ratio = 0.5
    fileFloders = ['concate', 'origin']
    path_base = PATH_BASE + '/' + data_class + "_" + data_date
    for data_group in data_groups:
        path = path_base + '/' + data_group
        concatenation = Concatenation(path, data_class)
        concatenation.getConcate(path, dict_concate[data_group], ratio)
        concatenation.getOrigin(path, dict_origin[data_group])
#    for fileFloder in fileFloders:
#        concatenation.addAll(path_base, data_groups, fileFloder)
    
if __name__ == '__main__':
    PATH_BASE   = "../../Data"
    run(PATH_BASE)   
