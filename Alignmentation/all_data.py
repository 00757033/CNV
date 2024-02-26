import shutil
import os
import csv




def copyfile(input_file,outputfile,layers = {"3":"OR","4":"CC"}):
    
    
    
if __name__ == '__main__':
    date = '0205'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    PATH_LABEL = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    image_path = PATH_BASE + 'inpaint/'
    PATH_MATCH = image_path + 'MATCH/' 
    PATH_MATCH_LABEL = image_path + 'MATCH_LABEL/' 