
from DataPreprocessing.getData2 import getData
from splitData import splitData
from relabels import reLabel
from preprocessing import PreprocessData
from AugData import Augment
# import sys
# import os
# # 取得 my_project 資料夾的絕對路徑
# my_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # 將 my_project 資料夾的路徑添加到 Python 解釋器的搜索路徑
# sys.path.append(my_project_path)

import numpy
import tools.tools as tools 


get_data = True
split_train_test = False
relabeled = False
preprocessed = False
augmented = False

date = '1120'
disease = 'PCV'
PATH = "../../Data/"
PATH_BASE =  PATH + "/" + disease + "_"+ date
PATH_LABEL = PATH + "/" + "new_label"
PATH_IMAGE = PATH + "/" + "OCTA"

DATA_GROUPS = ["OR","CC"]

def main(data_class,data_date):

    # new folder
    tools.makefolder(PATH + data_class + "_" + data_date)
    # Get Data 
    data = getData(PATH_BASE,PATH_IMAGE,PATH_LABEL)
    if get_data:
        print("Start get data...")       
        data.labelDict("label")
        g = data.getLabelDict()
        data.writeLabelJson()
        data.getData(PATH_BASE)
        data.getgroup()
        data.saveGroupDataframe(PATH + data_class + "_" + data_date + '/group.csv')
    else:
        data.getgroup()
        data.writeDataframe()

    if split_train_test :
        data_df = data.getDataframe()
        split = splitData(PATH_BASE,data_df)
        split.splitData()
        split.saveData()

    if relabeled :
        label= reLabel(PATH_BASE + '/' + 'trainset')
        # label.relabel('ROI_OTSU_contour')
        label.relabel2('OTSU_ROI_contour_RETR_CCOMP')
        label.origin_relabel('otsu')

    if preprocessed :
        preprocess = PreprocessData(PATH_BASE + '/' + 'trainset')
        preprocess.preprocess('otsu','otsu_UnsharpMask')

    if augmented:
        preprocess = Augment(PATH_BASE + '/' + 'trainset')
        times = 4
        preprocess.augumentation('otsu_UnsharpMask','otsu_UnsharpMask_aug' + str(times),times)   


if __name__ == '__main__':
    date = '0805'
    disease = 'PCV'
    main(disease,date)