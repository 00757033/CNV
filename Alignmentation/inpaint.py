import cv2
import numpy as np
import os
from pathlib import Path
import tools.tools as tools
import pandas as pd
import datetime 
import shutil
import pandas as pd

class inpaint():
    def __init__(self,path,output_path,layers = ["1","2","3","4"]):
        self.path = path
        self.points = []
        self.output_path = output_path
        self.layers = layers
        self.data_list = ['1','2', '3', '4']

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
        if os.path.exists('points.csv'):
            df.to_csv('points.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('points.csv', index=False)

    def extraneous_information(self,file,newpoint =False,point = 'points.csv'):
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
            # remove the unnecessary parts
            remove_img[self.y:self.y+self.height, self.x:self.x+self.width] = 0
            # cv2.imshow('remove_img', remove_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # inpainting
            remove_img = cv2.cvtColor(remove_img, cv2.COLOR_RGB2BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            dst = cv2.inpaint(remove_img, mask, 5, cv2.INPAINT_TELEA)
            window_size = 30
            dst = self.texture_synthesis(dst, self.y, self.x, self.height, self.width, 500, window_size)
            return dst
        else:
            print("Failed to load the image.")
            return
    def print_points(self):
        print("x: {}, y: {}, height: {}, width: {}".format(self.x, self.y, self.height, self.width))

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

    def all_inpaint(self):
        for data in self.data_list:
            tools.makefolder(self.output_path  + data)
            for file in Path(self.path + '/' + data).iterdir():
                if file.suffix == '.png':
                    # print(file)
                    dst = self.extraneous_information(str(file))
                    # print(self.path + '/inpaint/' + data +'/'+ file.name)
                    cv2.imwrite(self.output_path + data +'/'+ file.name, dst)
                    # print("Save the file: ", file)
                    # print("--------------------------------------------------")
                    # cv2.imshow('dst', dst)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
    def inject(self,file = '../../Data/打針資料.xlsx',label = ["診斷","病歷號","眼睛","打針前門診日期","三針後門診","六針後門診","九針後門診","十二針後門診"]):
        self.inject_df = pd.DataFrame()
        self.inject_df = pd.read_excel(file, sheet_name="20230830",na_filter = False, engine='openpyxl')
        print()
        self.inject_df['病歷號'] = self.inject_df['病歷號'].apply(lambda x: '{:08d}'.format(x))
        self.inject_df = self.inject_df.sort_values(by=["病歷號","眼睛"])
        columns_of_interest = ["病歷號","眼睛","打針前門診日期","三針後門診","六針後門診","九針後門診","十二針後門診"]
        self.inject_df = self.inject_df[columns_of_interest]
        eyes = {'OD':'R','OS':'L','OU':['R','L']}
        for data_name in self.inject_df["眼睛"].values:
            if data_name in eyes:
                self.inject_df["眼睛"].replace(data_name,eyes[data_name],inplace = True)
                

    def OCTA_inpaint(self,PATH,disease_folder):
        self.inject()
        for layer in self.data_list:
            print('layer',layer)
            tools.makefolder(self.path  + layer)
        self.all_data= PATH + '/OCTA'
        for patient in os.listdir(self.all_data):
            if patient in self.inject_df['病歷號'].values:
                for date in os.listdir(self.all_data + '/' + patient):
                    for eye in os.listdir(self.all_data + '/' + patient + '/' + date):
                        if eye in self.inject_df[self.inject_df['病歷號'] == patient]['眼睛'].values:
                            print('patient',patient,eye,date)
                            for layer in self.data_list:
                                if os.path.exists(self.all_data + '/' + patient + '/' + date + '/' + eye + '/' + layer + '.png'):
                                    dst = self.extraneous_information(self.all_data + '/' + patient + '/' + date + '/' + eye + '/' + layer + '.png')
                                    if dst is not None:
                                        
                                        # print(self.path  + '/'  + layer + '/'+ patient +'_' +eye +'_'+  date+ '.png')
                                        cv2.imwrite(self.path  + '/'  + layer + '/'+ patient +'_' +eye +'_'+  date+ '.png', dst)
                
        
if __name__ == "__main__":
    date = '0305'
    disease = 'PCV'
    PATH = "../../Data/"
    
    disease_folder = PATH + "/" + disease + "_"+ date
    IMAGE_PATH = disease_folder + "/" + "ALL"
    PATH_BASE =  disease_folder + '/inpaint/'
    preprocess = inpaint(IMAGE_PATH,PATH_BASE)
    preprocess.all_inpaint()
    # file1 = "..\\..\\Data\\OCTA\\00294362\\20210511\\L\\4.png"
    # file2 = "..\\..\\Data\\OCTA\\00294362\\20221222\\L\\3.png"
    # dst = preprocess.extraneous_information(file2)
    # preprocess.print_points()
    # preprocess.OCTA_inpaint(PATH,disease_folder)
