# tensorboard --logdir=./logs 
from trainModel3_eval import *
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
result= False
gradcam_signal = False
#data_class為病徵的名稱，如:PED, CNV
def main(data_class):
    #Date time
    data_date = "20240502"
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


    #HyperParameter
    image_size   = 304
    models       = ['UNetPlusPlus','DCUNet','BCDUNet','DenseUNet'] # 'UNet','UNetPlusPlus','R2UNet','MultiResUNet','DCUNet','BCDUNet' ,'DenseUNet','FRUNet','SDUNet'
    epochs       = [1] #50、100、200、400
    datas        = ['train']
    batchs       = [4]
    lrns         = [0.01] #0.01,0.001、0.0001
    filters      = [[32,64,128,256,512]] #[32,64,128,256,512], [16,32,64,128,256],

    #Evaluate prediction threshold
    predict_threshold = 0.5
    channel = 3
    #Get label data-------------------------------------------------------------------------------------------------
    if train_data:
        # make 
        tools.makefolder(PATH_DATASET)
        
        Train = train(data_class, data_date, PATH_MODEL, PATH_RESULT, image_size, models, batchs, epochs, datas, lrns, filters, predict_threshold,channel=channel)
        Train.run(PATH_DATASET, train_signal,postprocess_signal = True, gradcam_signal = gradcam_signal)
        # model_data = 'PCV_20240502_connectedComponent_bil51010_clah1016_concate34OCT_42_CC/train'
        # data = path_base + '/PCV_20240502_connectedComponent_bil51010_clah1016_concate34OCT_CC'
        # Train.get_best_model(model_data,data,'UNet',2,150,0.01,[32,64,128,256,512])
    #Result--------------------------------------------------------------------------------------------------------
    if result :
        PATH = './Result/' + data_class  + '_' + data_date
        getResult(PATH)






if __name__ == '__main__':
    main('PCV')
