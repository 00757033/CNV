import csv
import os
import json
import shutil

def copyfile(input_file,outputfile,layers = {"3":"OR","4":"CC"}):
    
    label = input_file.replace('MATCH','MATCH_LABEL')
    input_file = input_file
    for layer in layers:
        layer_input_file = input_file + layer+ '/'
        layer_output_file = outputfile + layers[layer] + '/' +'images/'
        label_input_file = label + layers[layer]+ '/'
        label_output_file = outputfile + layers[layer] + '/' +'masks/'
        # if not os.path.exists(layer_output_file):
        #     os.makedirs(layer_output_file, exist_ok=True)
        # if not os.path.exists(label_output_file):
        #     os.makedirs(label_output_file, exist_ok=True)
        shutil.copytree(layer_input_file, layer_output_file)
        shutil.copytree(label_input_file, label_output_file)
        # print('copy',layer_input_file, layer_output_file)
        # print('copy',label_input_file, label_output_file)
    
    



if __name__ == '__main__':

    date = '0205'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = disease + '_' + date
    file_name = PATH_DATA + PATH_BASE + '\ALL\inpaint\MATCH\crop_KAZE_BF_0.8\\'
    path = PATH_DATA +PATH_BASE +  '/compare/'
    copyfile(file_name,path)