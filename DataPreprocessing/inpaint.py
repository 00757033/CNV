import cv2
import numpy as np
import os
from pathlib import Path
import tools.tools as tools
import pandas as pd
import datetime 
import matplotlib.pyplot as plt

class inpaint():
    def __init__(self,path,layers = {"3":"OR","4":"CC"}):
        self.path = path
        self.points = []
        self.layers = layers

    def get_points(self, im,large = 4):
        # new a window and bind it to the callback function
        cv2.namedWindow('image')
        img = cv2.imread(im)
        self.img = img
        # crop the image
        img = img[img.shape[0]//2:img.shape[0], 0:img.shape[1]//2]
        # enlarge the image
        img = cv2.resize(img, (0, 0), fx=large, fy=large, interpolation=cv2.INTER_NEAREST)
        cv2.setMouseCallback('image', self.on_mouse, param=img)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.save_points(large)
        if self.points == []:
            print("No new points")
            return False
        return True
        
    def on_mouse(self, event, x, y, flags, param):
        # get mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            print("get points: (x, y) = ({}, {})".format(x, y))
            self.points.append((x, y))
            # draw line on the image
            cv2.line(param,(x,0),(x,param.shape[0]),(0,0,255),2)
            cv2.line(param,(0,y),(param.shape[1],y),(0,0,255),2)

            cv2.imshow('image', param)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("remove points: (x, y) = ({}, {})".format(x, y))
            if (x, y) in self.points:
                self.points.remove((x, y))
                # remove the circle
                cv2.rectangle(param, (x-3, y-3), (x+3, y+3), (255, 0, 0), 3)
                cv2.imshow('image', param)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No points in this position")

    def save_points(self,large = 4):
        min_x = min([x for x, y in self.points])
        max_x = max([x for x, y in self.points])
        min_y = min([y for x, y in self.points])
        max_y = max([y for x, y in self.points])
        self.x = min_x // large
        self.y = min_y // large + self.img.shape[0]//2
        self.height = (max_y - min_y) // large
        self.width = (max_x - min_x) // large
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.DataFrame({'time': time ,'x': self.x, 'y': self.y, 'height': self.height, 'width': self.width}, index=[0])

        # if the file exists, append the new data to the file
        if os.path.exists('./record/points.csv'):
            df.to_csv('./record/points.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('./record/points.csv', index=False)

    def extraneous_information(self,file,newpoint =False,point = './record/points.csv'):
        # remove unnecessary Information from the image
        data_dir = ['images','masks']
        # x = 7
        # y = 273
        # height = 24
        # width = 15
        check =True
        while True:
            if not os.path.exists(point) or newpoint :
                check = self.get_points(file)
            df = pd.read_csv(point)
            if df.empty:
                print("No points in this file")
                check = self.get_points(file)
            if check :
                break
        # get the last point
        self.x = df['x'].iloc[-1]
        self.y = df['y'].iloc[-1]
        self.height = df['height'].iloc[-1]
        self.width = df['width'].iloc[-1]

        # remove the unnecessary information
        img = cv2.imread(file)
        if img is  None:
            print("Failed to load the image.")
            return None
        img = cv2.resize(img,(304,304))
        
        if img is not None:
            mask = img.copy()
            remove_img = img.copy()
            # Leave the necessary parts and fill the rest with black
            mask[0:self.y, 0:img.shape[1]] = 0
            mask[self.y+self.height:img.shape[0], 0:img.shape[1]] = 0
            mask[0:img.shape[0], 0:self.x] = 0
            mask[0:img.shape[0], self.x+self.width:img.shape[1]] = 0
            self.mask = mask
            
            # plt.imshow(remove_img)
            # plt.axis('off')
            # plt.show()
            
            # plt.imshow(mask)
            # plt.axis('off')
            # plt.show()
            
            # remove the unnecessary parts
            remove_img[self.y:self.y+self.height, self.x:self.x+self.width] = 0
            
            # plt.imshow(remove_img)
            # plt.axis('off')
            # plt.show()
            

            # inpainting
            remove_img = cv2.cvtColor(remove_img, cv2.COLOR_RGB2BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            dst = cv2.inpaint(remove_img, mask,2, cv2.INPAINT_TELEA)
            window_size = 10
            dst = self.texture_synthesis(remove_img, self.y, self.x, self.height, self.width, 200, window_size)
            
            # plt.imshow(dst, cmap='gray')
            # plt.axis('off')
            # plt.show()
            return dst
        else:
            print("Failed to load the image.")
            return
    def print_points(self):
        print("x: {}, y: {}, height: {}, width: {}".format(self.x, self.y, self.height, self.width))
        
    def get_points(self):
        return self.x, self.y, self.height, self.width

    def texture_synthesis(self,image, y, x, height, width, iterations, window_size):
        best_mse = float('inf')  # 用于存储最佳均方误差
        for _ in range(iterations):
            # 随机选择一个窗口位置
            i, j = np.random.randint(0, image.shape[0] - window_size), np.random.randint(0, image.shape[1] - window_size)
            
            # 截取窗口和目标区域，确保它们的大小相同
            window = image[i:i+height, j:j+width]
            target_region = image[y:y+height, x:x+width]
            
            # 检查窗口和目标区域的形状
            if window.shape != target_region.shape:
                continue  # 如果形状不匹配，跳过此次迭代
            
            # 计算窗口与目标区域之间的均方误差
            mse = np.mean(np.square(window - target_region))
            
            # 在缺失区域中合成与选定窗口相似的纹理
            if mse < best_mse:
                best_mse = mse
                best_i, best_j = i, j
                image[y:y+height, x:x+width] = window
        
        return image
if __name__ == "__main__":
    date = '0205'
    disease = 'PCV'
    PATH = "../../Data/"
    PATH_BASE =  PATH + "/" + disease + "_"+ date
    preprocess = inpaint(PATH_BASE)
    file1 = "..\\..\\Data\\OCTA\\00294362\\20210511\\L\\4.png"
    file2 = "..\\..\\Data\\OCTA\\00294362\\20221222\\L\\3.png"
    dst = preprocess.extraneous_information(file2)
    preprocess.print_points()
    img = cv2.imread(file1)
    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.imwrite("../../Data/PPT/00294362_20221222_3.png", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()