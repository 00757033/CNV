import cv2
import numpy as np
import os
import shutil
def copy_file(input_img_path, input_mask_path, output_img_path, output_mask_path,layers= ["1","2","3","4"],label_layers={"4":"CC"}):
    print(input_img_path)
    for lay in os.listdir(input_img_path):
        if lay in layers:
            print(lay)
            source = os.path.join(input_img_path, lay)
            dest = os.path.join(output_img_path, lay)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
            if lay in label_layers:
                img_source = os.path.join(input_img_path, lay)
                img_dest = os.path.join(output_img_path, label_layers[lay],'images')
                if os.path.exists(img_dest):
                    shutil.rmtree(img_dest)
                shutil.copytree(os.path.join(input_img_path, lay), os.path.join(output_img_path, label_layers[lay],'images'))
                
                msk_source = os.path.join(input_mask_path, label_layers[lay])
                msk_dest = os.path.join(output_mask_path, label_layers[lay],'masks')
                if os.path.exists(msk_dest):
                    shutil.rmtree(msk_dest)
                    
                
                shutil.copytree(msk_source, msk_dest)
            



if __name__ == "__main__":
    date = '20240320'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'
    inpaint = 'inpaint/'
    mathods = 'crop_KAZE_FLANN_0.8'
    input_img_path = PATH_BASE + inpaint + 'MATCH/'+ mathods
    input_mask_path = PATH_BASE + inpaint + 'MATCH_LABEL/' + mathods
    output_img_path = PATH_BASE + 'align/' 
    output_mask_path = PATH_BASE + 'align/' 
    
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
        
    copy_file(input_img_path, input_mask_path, output_img_path, output_mask_path)
        
    
    