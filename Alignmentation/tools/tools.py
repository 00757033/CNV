import os
import json
import cv2
import numpy as np
def remove_exist_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)

def write_to_json_file(file_name, data):
    with open(file_name, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True, ensure_ascii=True)

# write patient to txt
def txt_to_file(set_patient,file_path = './',file_name = 'patient',title = False,line = False):
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    file = open(file_path+'/'+file_name+'.txt', 'a')

    if title:
        file.write(title+'\n')

    for new in set_patient: 
        # 8 bit for patient
        if len(new) != 8:
            new = '0'* (8-len(new)) + new
        
        file.write(new + '\n')

    file.write('total '+str(len(set_patient))   )

    # need to split
    if line:
        file.write('='*20 + '\n')

    file.close()

# write dictionary to txt
def dict_to_txt(data, file_path='./', file_name='patient', new_line=False, line=False):
    with open(file_path + '/' + file_name + '.txt', 'w') as file:
        _dict_to_txt(data, file, new_line, line)
        file.write('total ' + str(len(data)) + '\n')

def _dict_to_txt(data, file, new_line, line,indent = 0):
    for key ,value in data.items():
        if isinstance(value,dict):
            file.write('\t'*indent+key+' : '+ '\n')
            if new_line:
                file.write('\n')
            _dict_to_txt(value,file,new_line,line,indent+1)
        elif isinstance(value, list):
            file.write('\t' * indent + key + ' : ' + ', '.join(map(str, value)) + '\n')
        else:
            file.write('\t' * indent + key + ' : ' + str(value) + '\n')
    

def pop_empty_dict(data):
    key_to_remove = list()
    for key ,value in data.items():
        if isinstance(value,dict):
            pop_empty_dict(value)
        if not value or len(value)==0:
            key_to_remove.append(key)
    for key in key_to_remove:
        data.pop(key)
    return data   

def overlay_images(image1, image2, alpha=0.5):
    # Ensure image paths are valid strings
    # same size
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions.")

    # add weighted images
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(image1, alpha, image2, beta, 0)
    return overlay

def makefolder(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Convert float32 to float recursively
def convert_float32_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_float32_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(item) for item in obj]
    else:
        return obj
 
def format_json(str):
    resultStr = json.dumps(json.loads(str), indent=4, ensure_ascii=False)
    return resultStr

def writeDataframe(data,path,name):
        data.to_csv(path +'/' + name+'.csv',index = False)

def get_label_path(input_path, layer_name):
    if input_path:
        layer_path = f"{input_path}_{layer_name}"
    else:
        layer_path = layer_name
    return layer_path

