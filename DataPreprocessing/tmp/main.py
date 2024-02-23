from getlabelData import *
from ConcateData import *
from splitData import *
from trainModel import *
from getAugData import *
from preprocess import *
from relabelCNV import *
from result import *

#Control signel
get_label_data = True
preprocessing_data = True
get_concate_data = True
ostu_label = True
split_data = True
augment_data = True
train_model = True
train_signal = True

#data_class為病徵的名稱，如:PED, CNV
def main(data_class):

    #Date time
    data_date = "0819"
    #Path definition
    PATH_BASE = "../../Data/"
    PATH_LABEL = PATH_BASE + "Label"
    PATH_IMAGE = PATH_BASE + "OCTA"
    PATH_MODEL   =  './Model/'    
    PATH_RESULT  = '/Result/'
    path_base = PATH_BASE + '/' + data_class + "_" + data_date
    PATH_DATASET = path_base + '/' +'trainset'
    #Data layer 
    data_groups  = ["OR", "CC"]

    #要合併的OCTA、OCT影像
    if  data_class == 'PED':
        dict_concate = {'OR': ["3_OCT", "4_OCT", "3"] , 'CC': ["4_OCT", "3_OCT", "4"]}
        dict_origin = {'OR': "3_OCT" , 'CC': "4_OCT"}
    elif data_class == 'CNV':
        dict_concate = {'OR': ["3", "4", "otsu"] , 'CC': ["4", "3", "otsu"]}
        #dict_concate = {'OR': ["3", "4", "3_OCT"] , 'CC': ["4", "3", "4_OCT"]}
        dict_origin = {'OR': "3" , 'CC': "4"}

    #Concate or not
    input_floders = ['concate', 'origin']
    #concate ostu層 的數值的加權
    ratio = 0.5

    #label seelection
    label = 'label_otsu' #label_otsu

    #OR + CC = ALL
    add_all = False

    #Augment Time
    augment_times = [2, 5, 10]

    #HyperParameter
    image_size   = 304
    models       = ['UNet', 'AttentionUNet', 'BCDUNet', 'UNetPlusPlus'] #['UNet', 'AttentionUNet', 'BCDUNet', 'UNetPlusPlus'] 
    epochs       = [200] #50、100、200、400
    datas        = ['train'] #['train', 'augment5', 'augment10']
    batchs       = [2, 4, 8]
    lrns         = [0.0001]
    filters      = [32, 64, 128, 256, 512]

    #Evaluate prediction threshold
    predict_threshold = 0.5


    #Get label data------------------------------------------------------------------------------------------------
    #取得病徵(data_class)Label的影像以及對應的OCT跟OCTA影像 並存放在一起，命名方式為 "病徵" + "日期"
    makefolder(path_base)
    if get_label_data : 
        print("Start get label data : " + data_class )
        data = getLabelData(PATH_LABEL, PATH_IMAGE, path_base) #getLabelData.py
        data.getLabelData(data_class)

    #Preprocess data----------------------------------------------------------------------------------------------
    if preprocessing_data : 
        for data_group in data_groups:
            path_preprocess = path_base + '/' + data_group
            preprocess = PreprocessData()
            preprocess.preprocess(path_preprocess, dict_origin[data_group])

    #Get concate data----------------------------------------------------------------------------------------------
    if get_concate_data :
        for data_group in data_groups:
            path = path_base + '/' + data_group
            concatenation = Concatenation(path, data_class)
            concatenation.getConcate(path, dict_concate[data_group], ratio)
            concatenation.getOrigin(path, dict_origin[data_group])

    #Relabel CNV---------------------------------------------------------------------------------------------------
    #將label與otsu threshold取交集
    if ostu_label :
        mask_type ='label'
        if data_class == 'CNV':
            mask_type = 'label_otsu'
            data_path = PATH_BASE + data_class + '_' + data_date
            for data_group in data_groups:
                path = data_path + '/' + data_group
                relabeler(path)
    
    #Split data to : Train Test Valid--------------------------------------------------------------------------------
    if split_data :
        data_all = "ALL"
        mask_type = label
        for input_floder in input_floders:
            for data_group in data_groups:
                data_form = data_class  + '_' + input_floder + '_' + data_date
                output_floder = data_form + '_' + data_group
                splitdata = splitData(path_base, data_group)
                splitdata.splitTrainTest(input_floder, output_floder, mask_type) 

    #Augment data--------------------------------------------------------------------------------------------------
    if augment_data : 
        for dataset in glob.glob("%s/*" %PATH_DATASET):
            for augment_time in augment_times:
                PATH_DATASET_INPUT = dataset + '/' + 'train'
                PATH_DATASET_AUG   = dataset + '/' + 'augment' + str(augment_time)
                augment = Augment()
                augment.augmentData(PATH_DATASET_INPUT, PATH_DATASET_AUG, augment_time)

    #Train---------------------------------------------------------------------------------------------------------
    if train_model :
        makefolder(PATH_RESULT)
        makefolder(PATH_MODEL)
        Train = train(data_class, data_date, PATH_MODEL, PATH_RESULT, image_size, models, batchs, epochs, datas, lrns, filters, predict_threshold)
        Train.run(PATH_DATASET, train_signal)
    
    #Result--------------------------------------------------------------------------------------------------------
    PATH = 'C:/Users/user/Desktop/sharonliu/eyes/' + data_class  + '_' + data_date
    getResult(PATH)

if __name__ == '__main__':
    main('CNV')
