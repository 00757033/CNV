# tensorboard --logdir=./logs 
from trainModel3 import *
import tools.tools as tools
from postprocessing3 import *
from result2 import *
import os
import warnings


# Suppress NumPy deprecation warnings
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning:tensorflow"

# Suppress NumPy deprecation warnings
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
# Disable NumPy deprecation warnings for TensorFlow
os.environ['TF_NUMPY_BANNER'] = 'off'
#Get label data
train_signal = True
train_data = True
postprocess_signal = True
result= True
#data_class為病徵的名稱，如:PED, CNV
def main(data_class):
    #Date time
    data_date = "0205"
    #Path definition
    PATH_BASE = "../../Data/"
    PATH_LABEL = PATH_BASE + "Label"
    PATH_IMAGE = PATH_BASE + "OCTA"
    PATH_MODEL   =  './Model/'    
    PATH_RESULT  = './Result/'
    path_base = PATH_BASE + '/' + data_class + "_" + data_date
    PATH_DATASET = path_base + '/' +'trainset'
    #Data layer 
    data_groups  = ["OR", "CC"]

    #要合併的OCTA、OCT影像

    if data_class == 'CNV':
        dict_concate = {'OR': ["3", "4","otsu"] , 'CC': ["4", "3","otsu"]}
        #dict_concate = {'OR': ["3", "4", "3_OCT"] , 'CC': ["4", "3", "4_OCT"]}
        dict_origin = {'OR': "3" , 'CC': "4"}

    #HyperParameter
    image_size   = 304
    models       = ['UNet','UNetPlusPlus','AttUNet','R2UNet','DenseUNet','MultiResUNet','DCUNet','FRUNet','BCDUNet'] # 
    epochs       = [150] #50、100、200、400
    datas        = ['train']
    batchs       = [2,4,8]
    lrns         = [0.001]
    filters      = [16,32, 64, 128, 256]

    #Evaluate prediction threshold
    predict_threshold = 0.5

    #Get label data-------------------------------------------------------------------------------------------------
    if train_data:
        # make 
        tools.makefolder(PATH_DATASET)
        
        Train = train(data_class, data_date, PATH_MODEL, PATH_RESULT, image_size, models, batchs, epochs, datas, lrns, filters, predict_threshold)
        Train.run(PATH_DATASET, train_signal,postprocess_signal)

    #Result--------------------------------------------------------------------------------------------------------
    if result :
        PATH = './Result/' + data_class  + '_' + data_date
        getResult(PATH)






if __name__ == '__main__':
    main('PCV')
