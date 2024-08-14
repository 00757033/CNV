import cv2
import numpy as np
import os
from pathlib import Path
import tools.tools as tools
import pandas as pd
import datetime 
import shutil
import pandas as pd


def copy_data(disease_folder,input_data , output_path , image_file= 'images_original', mask_file = 'predict'):
    print(disease_folder + '/'+  input_data + '/images_original', PATH_BASE + output_path + '/images_original')
    # shutil.copytree(disease_folder + input_data + '/images_original', PATH_BASE + output_path + '/images_original')
    # shutil.copytree(disease_folder + input_data + '/predict', PATH_BASE + output_path + '/predict')
    
    
    
    
    
if __name__ == '__main__':
    date = '20240411'
    disease = 'PCV'
    PATH = "../../Data/"
    
    disease_folder = PATH + "/" + disease + "_"+ date
    IMAGE_PATH = disease_folder + "/" + "ALL"
    PATH_BASE =  disease_folder + '/inpaint/'
    
    input_data = 'PCV_20240418_connectedComponent_bil51010_clah1016_concate34OCT_CC'
    output_path = 'generate_data'
    copy_data(disease_folder,input_data , output_path , image_file= 'images_original', mask_file = 'predict')
    
    