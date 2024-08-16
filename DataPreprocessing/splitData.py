from sklearn.model_selection import train_test_split

import tools.tools as tools
import os
import cv2
import shutil

class splitData():
    def __init__(self, path,layers = {"4":"CC"}):
        self.train_test = ["train","test","valid"]
        self.path = path
        self.df = None
        self.train_df = None
        self.test_df= None
        self.valid_df= None
        self.layers = layers

    def splitData(self,name,output_name,random = 42):
        tools.makefolder(os.path.join(self.path, "trainset"))
        print("start split data")
        random_state = str(random)
        for layer in self.layers:
            for i in self.train_test:
                tools.makefolder(os.path.join(self.path, "trainset",output_name  + '_'+ self.layers[layer], i))
                tools.makefolder(os.path.join(self.path, "trainset",output_name  + '_'+ self.layers[layer], i, "images"))
                tools.makefolder(os.path.join(self.path, "trainset",output_name  + '_'+ self.layers[layer], i, "masks"))
                if 'images_original' in os.listdir(os.path.join(self.path, name  + '_'+ self.layers[layer])):
                    tools.makefolder(os.path.join(self.path, "trainset",output_name  + '_'+ self.layers[layer], i, "images_original"))

        train_ratio = 0.7  # 訓練集比例
        test_ratio = 0.2 # 測試集比例
        val_ratio = 0.1   # 驗證集比例

        # 確保比例總和不大於1.0
        assert train_ratio + test_ratio + val_ratio <= 1.0


        for layer in self.layers:
            print("split",self.layers[layer])
            path_input = os.path.join(self.path, name  + '_'+ self.layers[layer])
            path_output = os.path.join(self.path, "trainset", output_name  + '_'+ self.layers[layer])

            # 取得資料夾內的所有檔案列表
            all_files = os.listdir(os.path.join(path_input, "images"))

            # 切分資料集
            print("split data")
            print("all_files",os.path.join(path_input, "images"),len(all_files))
            
            # 切分資料集
            train_test_data, val_data = train_test_split(all_files, test_size=val_ratio, random_state=random)
            train_data, test_data = train_test_split(train_test_data, test_size=test_ratio/(train_ratio+test_ratio), random_state=random)

            print("training data:",len(train_data))
            print("vaild data:",len(val_data))
            print("testing data:",len(test_data))

            # 建立資料集
            self.create_dataset(path_input, path_output, train_data, "train")
            self.create_dataset(path_input, path_output, test_data, "test")
            self.create_dataset(path_input, path_output, val_data, "valid")
            if 'images_original' in os.listdir(os.path.join(self.path, name  + '_'+ self.layers[layer])):
                for file in train_data:
                    shutil.copy(os.path.join(path_input, "images_original", file), os.path.join(path_output, "train", "images_original", file))
                    
                for file in test_data:
                    shutil.copy(os.path.join(path_input, "images_original", file), os.path.join(path_output, "test", "images_original", file))
                    
                for file in val_data:
                    shutil.copy(os.path.join(path_input, "images_original", file), os.path.join(path_output, "valid", "images_original", file))
                    

    def create_dataset(self, path_input, path_output, files, dataset_type):
        for file in files:
            # 複製檔案
            shutil.copy(os.path.join(path_input, "images", file), os.path.join(path_output, dataset_type, "images", file))
            shutil.copy(os.path.join(path_input, "masks", file), os.path.join(path_output, dataset_type, "masks", file))
            
            
            


if __name__ == "__main__":
    path = "../../Data"
    # date = '20240524'
    date = '20240524' # 20240524  20240525:original
    disease = 'PCV'
    NAME = disease + "_" + date
    path_base =  path + "/" + disease + "_"+ date
    image_path  = path + "/" + "OCTA"
    output_path = path_base 

    split = splitData(path_base)
    # split.splitData(NAME + '_connectedComponent' ,NAME + '_connectedComponent_30',random = 30)
    # split.splitData(NAME + '_connectedComponent_bil31010_clah0712_concate34' ,NAME + '_connectedComponent_bil31010_clah0712_concate34_30',random = 30)

    # split.splitData(NAME + '_connectedComponent_bil31010_clah0712_concate34OCT' ,NAME + '_connectedComponent_bil31010_clah0712_concate34OCT_30',random = 30)


    split.splitData(NAME + '_connectedComponent_bil31010_clah0712_concate34OCT' ,NAME + '_connectedComponent_bil31010_clah0712_concate34OCT_30',random = 30)