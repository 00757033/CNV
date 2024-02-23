from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import pandas as pd
import shutil
import os
# import sys
# # 取得 my_project 資料夾的絕對路徑
# my_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # 將 my_project 資料夾的路徑添加到 Python 解釋器的搜索路徑
# sys.path.append(my_project_path)
import tools.tools as tools 

class splitData():
    def __init__(self, path,data_df,layers = {"3":"OR","4":"CC"}):
        self.train_test = ["train","test","valid"]
        self.path = path
        self.data_df = data_df
        self.layers = layers
        self.df = None
        self.train_df = None
        self.test_df= None
        self.valid_df= None 

    def splitType(self,name = 'type'):
        df_type = self.data_df.groupby([name])
        return df_type

    def splitData(self,input_name,output_name):
        print("start split data")
        data = self.splitType()
        group_shuffle_split = GroupShuffleSplit(n_splits = 2, test_size = 0.1, random_state = 42)
        group_train_vail_split = GroupShuffleSplit(n_splits = 2, test_size = 0.125, random_state = 42)

        for type_name , group_type_df in data:
            train_valid_idx ,test_idx =next(group_shuffle_split.split(group_type_df, groups=group_type_df['group']))
            train_valid_subset = group_type_df.iloc[train_valid_idx]

            train_idx,valid_idx =next(group_train_vail_split.split(train_valid_subset, groups=train_valid_subset['group']))

            train_subset = train_valid_subset.iloc[train_idx]
            set_train = set(train_subset['group'])
            self.saveData(train_subset ,type_name[0],input_name,output_name, output_file_name = 'train')
            valid_subset = train_valid_subset.iloc[valid_idx]
            set_valid = set(valid_subset['group'])
            self.saveData(valid_subset ,type_name[0],input_name,output_name, output_file_name = 'valid')
            test_subset = group_type_df.iloc[test_idx]
            set_test = set(test_subset['group'])
            self.saveData(test_subset ,type_name[0],input_name,output_name, output_file_name = 'test')

            print("split",type_name[0])
            print("training data:",len(set_train),len(train_subset))
            print("vaild data:",len(set_valid),len(valid_subset))
            print("testing data:",len(set_test),len(test_subset))
           
    def saveData(self,data,layer ,input_name,output_name,output_file_name):
        data_dir = ['images','masks']
        input_layer_path = tools.get_label_path(input_name, layer)
        output_layer_path = tools.get_label_path(output_name, layer)
        input_dir = os.path.join(self.path,  input_layer_path)
        output_dir = os.path.join(self.path, "trainset", output_layer_path, output_file_name)
        for  dir in data_dir:
            tools.makefolder(os.path.join(output_dir,dir ))
        
        for image in data['image']:
            for  dir in data_dir:
                input_path = os.path.join(input_dir,  dir,image + '.png')
                output_path = os.path.join(output_dir,  dir,image + '.png')
                shutil.copy(input_path,output_path)
            # img_origin_path = 
            # mask_origin_path = img_path.replace('images','masks')
            # print('origin',self.path + '/' + layer_path + '/' + 'images' + '/' + image + '.png')
            # print('dest',self.path + '/trainset' + '/' + layer_path + '/' + output_file_name + '/' + 'images' + '/' + image + '.png')
            # shutil.copy(self.path + '/' + layer_path + '/' + 'images' + '/' + image + '.png',self.path + '/trainset' + '/' + layer_path + '/' + 'train' + '/' + 'images' + '/' + image + '.png')
            # shutil.copy(self.path + '/' + layer_path + '/' + 'masks' + '/' + image + '.png',self.path + '/trainset' + '/' + layer_path + '/' + 'train' + '/' + 'masks' + '/' + image + '.png')

           

if __name__ == "__main__":
    path = "../../Data"
    date = '0918'
    disease = 'PCV'
    file = disease + "_"+ date
    path_base =  path + "/" + disease + "_"+ date
    image_path  = path + "/" + "OCTA"
    output_path = path_base 

    data_df  = pd.read_csv('./record/group.csv')
    print(data_df)
    split = splitData(path_base,data_df)
    split.splitData('PCV_0918_otsu_bil_clahe','PCV_0918_otsu_bil_clahe_group_42')
