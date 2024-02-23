import cv2
import numpy as np
import pathlib as pl
from matplotlib import pyplot as plt
import time
import datetime
from pathlib import Path
import math
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import tools.tools as tools
import pandas as pd
import os
from skimage.filters import unsharp_mask
# clahe
# median filtering
# normalize

class PreprocessData():
    def __init__(self,path,layers = {"3":"OR","4":"CC"}):
        self.path = path
        self.layers = layers
        self.train_test = ["train","test","valid"]
        self.friqa = dict()

    def evaluate(self,image ,original_image):
        mse = mean_squared_error(image, original_image)
        psnr = peak_signal_noise_ratio(image, original_image)
        ssim = structural_similarity(image, original_image)
        return mse,psnr,ssim
    
    def friqa_dict(self,eval = ['mse','psnr','time'] ):
        friqa = dict()
        for name in eval :
            friqa[name] = dict()
        return friqa
    
    def filter_preprocess(self,input_path,output_name,save_img =True,new_parameter = False,path = './record/'):
        
        for layer in self.layers : 
            output_image, output_label = make_output_path(self.path,output_name, self.layers[layer])
            print(self.layers[layer])
            label_path = tools.get_label_path(input_path,self.layers[layer])
            file = path + self.layers[layer] + '_filter_parameter.csv'
            if new_parameter or not os.path.exists(file) :
                [d,sigmaColor,sigmaSpace] = self.filter_parameter(label_path)
                self.save_filter_parameter(input_path,self.layers[layer],[d,sigmaColor,sigmaSpace])

            d,sigmaColor,sigmaSpace = self.get_filter_parameter(input_path,self.layers[layer])
            # not return none
            if d == None or sigmaColor == None or sigmaSpace == None:
                [d,sigmaColor,sigmaSpace] = self.filter_parameter(label_path)
                self.save_filter_parameter(input_path,self.layers[layer],[d,sigmaColor,sigmaSpace])
            d,sigmaColor,sigmaSpace = self.get_filter_parameter(input_path,self.layers[layer])
            print('filter_parameter',d,sigmaColor,sigmaSpace)
            if save_img:
                for image in pl.Path(self.path+ '/' +label_path + '/'+ 'images').iterdir():
                    if image.suffix == '.png':
                        image_name = image.name
                        img_path = str(image)
                        gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        mask_path = img_path.replace('images','masks')
                        image_label = cv2.imread(mask_path, 0)
                        
                        bilateral_image = cv2.bilateralFilter(gray_image, d, sigmaColor,sigmaSpace )
                        cv2.imwrite(output_image + '/'+image_name,bilateral_image)
                        cv2.imwrite(output_label + '/'+image_name,image_label)

    def save_filter_parameter(self,input_name,layer_name,parameter,path = './record/'):
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.DataFrame({'file_name' : input_name,'time':time,'d':parameter[0],'sigmaColor':parameter[1],'sigmaSpace':parameter[2]},index=[0])
        if os.path.exists(path + layer_name + '_filter_parameter.csv') and os.path.getsize(path + layer_name + '_filter_parameter.csv') > 0:
            df.to_csv(path + layer_name + '_filter_parameter.csv',mode='a',header=False,index=False)
        else:
            df.to_csv(path + layer_name + '_filter_parameter.csv',index=False)

    def get_filter_parameter(self,input_name,layer_name,path = './record/'):
        df = pd.read_csv(path + layer_name + '_filter_parameter.csv')
        if df.empty:
            print('empty')
            return None, None, None
        elif input_name in df['file_name'].values:
            return df['d'].iloc[-1],df['sigmaColor'].iloc[-1],df['sigmaSpace'].iloc[-1]
        else:
            print('not in')
            return None, None, None

    def save_clahe_parameter(self,input_name,layer_name,parameter,path = './record/'):
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.DataFrame({'file_name' : input_name,'time':time,'clip':parameter[0],'kernel':parameter[1]},index=[0])
        if os.path.exists(path + layer_name + '_clahe_parameter.csv') and os.path.getsize(path +layer_name + '_clahe_parameter.csv') > 0:
            df.to_csv(path + layer_name + '_clahe_parameter.csv',mode='a',header=False,index=False)
        else:
            df.to_csv(path + layer_name + '_clahe_parameter.csv',index=False)

    def get_clahe_parameter(self,input_name,layer_name,path = './record/'):
        df = pd.read_csv(path + layer_name + '_clahe_parameter.csv')
        if df.empty:
            print('empty')
            return None, None
        elif input_name in df['file_name'].values:
            return df['clip'].iloc[-1],df['kernel'].iloc[-1]
        else:
            print('not in')
            return None , None
           
    def all_parameter(self,layer,save_para = True,file_path = './record/'):
        file = file_path + layer + '_' + 'all_parameter' + '.csv'
        best_mse = 1000000
        best_psnr = 0
        best_mse_parameter = None
        for kernel in range(3,20,2):
            for l in range(5,20,1):
                clip = l/10.0
                for i in range(5,10,1):
                    for j in range(10,40,5): # 10 - 200
                        for k in range(10,40,5): # 10 - 200
                            execution_time = 0
                            mse_list = []
                            psnr_list = []     
                            for image in Path( self.path+ '/' + layer+ '/'+ 'images').iterdir():   
                                if image.suffix == '.png':
                                    gray = cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
                                    start_time = time.time()  
                                    bilateral_image = cv2.bilateralFilter(gray, i, j,k )
                                    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(kernel,kernel))
                                    cl = clahe.apply(bilateral_image)
                                    end_time = time.time()
                                    mse,psnr,ssim = self.evaluate(cl,gray)
                                    execution_time += (end_time - start_time)
                                    mse_list.append(mse)
                                    psnr_list.append(psnr)

                            avg_mse = np.mean(mse_list).round(5)
                            avg_psnr = np.mean(psnr_list).round(5)
                            if avg_mse < best_mse:
                                best_mse = avg_mse
                                best_mse_parameter = [i,j,k,clip,kernel]
                            if avg_psnr > best_psnr:
                                best_psnr = avg_psnr
                                best_psnr_parameter = [i,j,k,clip,kernel]
                            if save_para:
                                df = pd.DataFrame({'layer':layer,'d':i,'sigmaColor':j,'sigmaSpace':k,'clip':clip,'kernel':kernel,'mse':avg_mse,'psnr':avg_psnr,'time':execution_time},index=[0])
                                if os.path.exists(file) and os.path.getsize(file) > 0:
                                    df.to_csv(file,mode='a',header=False,index=False)
                                else:
                                    df.to_csv(file,index=False)
        return best_psnr_parameter

    # # get all parameter
    # def preprocess(self,input_path,output_name,save_img =True,new_parameter = False,path = './record/'):
    #     for layer in self.layers : 
    #         output_image, output_label = make_output_path(self.path,output_name, self.layers[layer])
    #         print(self.layers[layer])
    #         label_path = tools.get_label_path(input_path,self.layers[layer])
    #         if new_parameter or not os.path.exists(path + self.layers[layer] + '_all_parameter.csv'):
    #             [d,sigmaColor,sigmaSpace,clip,kernel] = self.all_parameter(label_path)
    #             self.save_all_parameter(self.layers[layer],[d,sigmaColor,sigmaSpace,clip,kernel])
    #         d,sigmaColor,sigmaSpace,clip,kernel = self.get_all_parameter(input_path,self.layers[layer])
    #         # not return none
    #         if d == None or sigmaColor == None or sigmaSpace == None or clip == None or kernel == None:
    #             [d,sigmaColor,sigmaSpace,clip,kernel] = self.all_parameter(label_path)
    #             self.save_all_parameter(input_path,self.layers[layer],[d,sigmaColor,sigmaSpace,clip,kernel])
    #         d,sigmaColor,sigmaSpace,clip,kernel = self.get_all_parameter(input_path,self.layers[layer])
    #         print('all_parameter',d,sigmaColor,sigmaSpace,clip,kernel)
    #         if save_img:
    #             for image in pl.Path(self.path+ '/' +label_path + '/'+ 'images').iterdir():
    #                 if image.suffix == '.png':
    #                     image_name = image.name
    #                     img_path = str(image)
    #                     gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #                     mask_path = img_path.replace('images','masks')
    #                     image_label = cv2.imread(mask_path, 0)
                        
    #                     bilateral_image = cv2.bilateralFilter(gray_image, d, sigmaColor,sigmaSpace )
    #                     clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(kernel,kernel))
    #                     cl = clahe.apply(bilateral_image)

    #                     cv2.imwrite(output_image + '/'+image_name,cl)
    #                     cv2.imwrite(output_label + '/'+image_name,image_label)

    def preprocess(self,input_path,output_name,parm = { "3": [5,10,10,0.7,3],"4":[5,10,10,0.7,12]},fastNlMeansDenoising=True ,bilateralFilter =True ,Unsharp = True,clahe =True,   save_img =True,new_parameter = False,path = './record/'):
        for layer in self.layers : 
            output_image, output_label = make_output_path(self.path,output_name, self.layers[layer])
    
            label_path = tools.get_label_path(input_path,self.layers[layer])
            print('input_path',self.path+ '/' +label_path + '/'+ 'images')
            d,sigmaColor,sigmaSpace,clip,kernel = parm[layer]
            
            # not return none
            print('all_parameter',d,sigmaColor,sigmaSpace,clip,kernel)
            if save_img:
                for image in pl.Path(self.path+ '/' +label_path + '/'+ 'images').iterdir():
                    if image.suffix == '.png':
                        image_name = image.name
                        img_path = str(image)
                        gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        mask_path = img_path.replace('images','masks')
                        image_label = cv2.imread(mask_path, 0)
                        gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        preprocessing_image = gray_image.copy()
                        
                                                     # Defining the kernel to be used in Top-Hat 
                        # preprocessing_image2 = preprocessing_image.copy()

                        # filterSize =(3, 3) 
                        # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, 
                        #                                 filterSize) 
                        # tophat_img = cv2.morphologyEx(preprocessing_image2,  
                        #         cv2.MORPH_TOPHAT, 
                        #         kernel2) 
                        # fig , ax =  plt.subplots(2,2)
                        # print(image_name)
                        # ax[0][0].imshow(preprocessing_image2,cmap='gray')
                        # ax[0][1].imshow(tophat_img,cmap='gray')
                        
                        # plt.show()

                        if  fastNlMeansDenoising:
                            # Apply non-local means denoising
                            preprocessing_image = cv2.fastNlMeansDenoising(preprocessing_image, None, 7, 7,21)

                
                        if bilateralFilter : 
                            preprocessing_image = cv2.bilateralFilter(preprocessing_image, d, sigmaColor,sigmaSpace )
                            
                        if Unsharp :
                            preprocessing_image = cv2.GaussianBlur(preprocessing_image,(5,5),0)
                            preprocessing_image = cv2.addWeighted(gray_image,1.5, preprocessing_image,-0.5,0)

                        if clahe :
                            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(kernel,kernel))
                            preprocessing_image = clahe.apply(preprocessing_image)
                            # denoised_img_image = clahe.apply(denoised_img)
                            
                        # fig , ax =  plt.subplots(2,2)
                        # print(image_name)
                        # ax[0][0].imshow(gray_image,cmap='gray')
                        # ax[0][1].imshow(denoised_img,cmap='gray')
                        # ax[1][0].imshow(preprocessing_image,cmap='gray')
                        # ax[1][1].imshow(denoised_img_image,cmap='gray')
                        
                        # plt.show()
                            
                            #  # Defining the kernel to be used in Top-Hat 
                            # preprocessing_image2 = preprocessing_image.copy()

                            # filterSize =(3, 3) 
                            # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, 
                            #                                 filterSize) 
                            # tophat_img = cv2.morphologyEx(preprocessing_image2,  
                            #         cv2.MORPH_BLACKHAT, 
                            #         kernel2) 

                            # cv2.imshow("original",preprocessing_image2) 
                            # cv2.imshow("tophat", tophat_img) 
                            # cv2.waitKey(5000) 
                            # cv2.destroyAllWindows()

                            
                        cv2.imwrite(output_image + '/'+image_name,preprocessing_image)
                        cv2.imwrite(output_label + '/'+image_name,image_label)
    def save_all_parameter(self,input_name,layer_name,parameter,path = './record/'):
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.DataFrame({'file_name' : input_name,'time':time,'d':parameter[0],'sigmaColor':parameter[1],'sigmaSpace':parameter[2],'clip':parameter[3],'kernel':parameter[4]},index=[0])
        if os.path.exists(path + layer_name + '_all_parameter.csv') and os.path.getsize(path + layer_name + '_all_parameter.csv') > 0:
            df.to_csv(path + layer_name + '_all_parameter.csv',mode='a',header=False,index=False)
        else:
            df.to_csv(path + layer_name + '_all_parameter.csv',index=False)

    def get_all_parameter(self,input_name,layer_name,path = './record/'):
        df = pd.read_csv(path + layer_name + '_all_parameter.csv')
        if df.empty:
            print('empty')
            return None, None, None, None, None
        elif input_name in df['file_name'].values:
            return df['d'].iloc[-1],df['sigmaColor'].iloc[-1],df['sigmaSpace'].iloc[-1],df['clip'].iloc[-1],df['kernel'].iloc[-1]
        else:
            print('not in')
            return None, None, None, None, None

    def clahe_preprocess(self,input_path,output_name,save_img =True,new_parameter = False,path = './record/'):
        for layer in self.layers : 
            output_image, output_label = make_output_path(self.path,output_name, self.layers[layer])
            print(self.layers[layer])
            label_path = tools.get_label_path(input_path,self.layers[layer])
            if new_parameter or not os.path.exists(path + self.layers[layer] + '_clahe_parameter.csv'):
                [clip , kernel]  = self.clahe_parameter(label_path)
                self.save_clahe_parameter(self.layers[layer],[clip , kernel])
            clip , kernel = self.get_clahe_parameter(input_path,self.layers[layer])
            # not return none
            if clip == None or kernel == None:
                [clip , kernel]  = self.clahe_parameter(label_path)
                self.save_clahe_parameter(input_path,self.layers[layer],[clip , kernel])
            clip , kernel = self.get_clahe_parameter(input_path,self.layers[layer])
            print('clahe_parameter',clip , kernel )
            if save_img:
                for image in pl.Path(self.path+ '/' +label_path + '/'+ 'images').iterdir():
                    if image.suffix == '.png':
                        image_name = image.name
                        img_path = str(image)
                        gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        mask_path = img_path.replace('images','masks')
                        image_label = cv2.imread(mask_path, 0)
                        
                        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(kernel,kernel))
                        cl = clahe.apply(gray_image)

                        cv2.imwrite(output_image + '/'+image_name,cl)
                        cv2.imwrite(output_label + '/'+image_name,image_label)

    def destripe(self,input_path,output_name):
        for layer in self.layers : 
            output_image, output_label = make_output_path(self.path,output_name, self.layers[layer])
            print(self.layers[layer])
            label_path = tools.get_label_path(input_path,self.layers[layer])
            # 
            for image in pl.Path(self.path+ '/' +label_path + '/'+ 'images').iterdir():
                if image.suffix == '.png':
                    image_name = image.name
                    img_path = str(image)
                    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    mask_path = img_path.replace('images','masks')
                    image_label = cv2.imread(mask_path, 0)
                    new_img = self.destripe_octa_image(gray_image)
                    cv2.imwrite(output_image + '/'+image_name,new_img)
                    cv2.imwrite(output_label + '/'+image_name,image_label)

    def filter_parameter(self,layer,save_para = True,file_path = './record/'):
        file_name = file_path + layer + '_' + 'bilateralFilter' + '.csv'
        # bilateralFilter
        best_mse = 1000000
        best_psnr = 0
        best_mse_parameter = None
        for i in range(5,20,10):
            for j in range(10,20,5): # 10 - 200
                for k in range(10,20,5): # 10 - 200
                    # go through all the images
                    execution_time = 0
                    mse_list = []
                    psnr_list = []
                    for image in Path( self.path+ '/' + layer+ '/'+ 'images').iterdir():
                        if image.suffix == '.png':
                            gray = cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
                            start_time = time.time()  
                            bilateral_image = cv2.bilateralFilter(gray, i, j,k )
                            end_time = time.time()  
                            mse,psnr,ssim = self.evaluate(bilateral_image,gray)
                            execution_time += (end_time - start_time)  
                            mse_list.append(mse)
                            psnr_list.append(psnr)

                    avg_mse = np.mean(mse_list).round(5)
                    avg_psnr = np.mean(psnr_list).round(5)
                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_mse_parameter = [i,j,k]
                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        best_psnr_parameter = [i,j,k]
                    if save_para:
                        df = pd.DataFrame({'layer':layer,'d':i,'sigmaColor':j,'sigmaSpace':k,'mse':avg_mse,'psnr':avg_psnr,'time':execution_time},index=[0])
                        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                            df.to_csv(file_name,mode='a',header=False,index=False)
                        else:
                            df.to_csv(file_name,index=False)
                    
        return best_psnr_parameter
                  
    def clahe_parameter(self,layer,save_para = True,file_path = './record/'):  
        file_name = file_path + layer + '_' + 'clahe' + '.csv'
        # clahe
        best_mse = 1000000
        best_psnr = 0
        best_mse_parameter = None
        for i in range(5,8,1):
            clip = i/10.0
            for kernel in range(3,16,2):
                # go through all the images
                execution_time = 0
                mse_list = []
                psnr_list = []       
                for image in Path( self.path+ '/' + layer+ '/'+ 'images').iterdir():   
                    if image.suffix == '.png':
                        gray = cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
                        start_time = time.time()  
                        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(kernel,kernel))
                        cl = clahe.apply(gray)
                        end_time = time.time()  
                        mse,psnr,ssim = self.evaluate(cl,gray)
                        execution_time += (end_time - start_time)  
                        mse_list.append(mse)
                        psnr_list.append(psnr)

                avg_mse = np.mean(mse_list).round(5)
                avg_psnr = np.mean(psnr_list).round(5)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_mse_parameter = [clip,kernel]
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_psnr_parameter = [clip,kernel]
                if save_para:
                    df = pd.DataFrame({'layer':layer,'clip':clip,'kernel':kernel,'mse':avg_mse,'psnr':avg_psnr,'time':execution_time},index=[0])
                    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                        df.to_csv(file_name,mode='a',header=False,index=False)
                    else:
                        df.to_csv(file_name,index=False)
        return best_psnr_parameter      

    def sharp_parameter(self,layer):
        friqa = dict()
        eval = ['mse','psnr']
        for image in Path( self.path+ '/' + layer+ '/'+ 'images').iterdir():
            if image.suffix == '.png':
                image_name = image.name
                img_path = str(image)
                gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
                friqa['mse'] = dict()
                friqa['psnr'] = dict()
                for i in range(0.1,1,0.1):
                    if i not in friqa:
                        friqa[i] = dict()
                    for kernel in range(3,10,2):
                        sharp = self.sharp(gray_image,kernel,i)
                        mse,psnr,ssim = self.evaluate(sharp,gray_image)
                        if i not in friqa['mse']:
                            friqa['mse'][i] = dict()
                        if kernel not in friqa['mse'][i]:
                            friqa['mse'][i][kernel] = []
                        if i not in friqa['psnr']:
                            friqa['psnr'][i] = dict()
                        if kernel not in friqa['psnr'][i]:
                            friqa['psnr'][i][kernel] = []
                        friqa['mse'][i][kernel].append(mse)
                        friqa['psnr'][i][kernel].append(psnr)


        best_mse = 100000000
        best_psnr = 0
        best_mse_key = None
        best_psnr_key = None
        for i in friqa['mse']:
            for kernel in friqa['mse'][i]:
                friqa['mse'][i][kernel] = np.mean(friqa['mse'][i][kernel]).round(3)
                friqa['psnr'][i][kernel] = np.mean(friqa['psnr'][i][kernel]).round(3)
                if friqa['mse'][i][kernel] < best_mse:
                    best_mse = friqa['mse'][i][kernel]
                    best_mse_key = [i,kernel]
                if friqa['psnr'][i][kernel] > best_psnr:
                    best_psnr = friqa['psnr'][i][kernel]
                    best_psnr_key = [i,kernel]
        print('sharp_parameter')
        print('best mse',best_mse,best_mse_key)
        print('best psnr',best_psnr,best_psnr_key)

    def sharp(self,image,kernel_size = 3, strength = 0.5):
        # 高斯滤波
        gaussian = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # 拉普拉斯滤波
        laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)

        # 转换为8位无符号整数类型
        filtered_image = np.uint8(np.absolute(gaussian))

        # 原图像加上锐化后的图像，以增强细节
        sharpened_image = cv2.addWeighted(image, 1.0, filtered_image, strength, 0)

        return sharpened_image

    def destripe_octa_image(self,octa_image):
        # Convert OCTA image to floating point for processing
        octa_image = octa_image.astype(np.float32)

        # Apply a bilateral filter to the OCTA image to smooth out noise while preserving edges
        # smoothed_image = cv2.bilateralFilter(octa_image, -1, sigma_s, sigma_r)


        # Calculate the mean intensity profile along the A-scan (vertical) direction
        mean_profile = np.mean(octa_image, axis=1)

        # Subtract the mean profile from each A-scan to remove horizontal stripes
        destriped_image = octa_image - mean_profile[:, np.newaxis]

        # Calculate the mean intensity profile along the B-scan (horizontal) direction
        mean_profile = np.mean(destriped_image, axis=0)

        # Subtract the mean profile from the entire image to remove vertical stripes
        destriped_image -= mean_profile

        # Clip negative values to ensure the image remains non-negative
        destriped_image[destriped_image < 0] = 0

        # Normalize the destriped image to the range [0, 255] for display
        destriped_image = (destriped_image / np.max(destriped_image)) * 255

        # Convert the destriped image back to uint8 for display
        destriped_image = destriped_image.astype(np.uint8)

        return destriped_image

def make_output_path(path ,output_name, layer_name):
        output_image = os.path.join(path, f"{output_name}_{layer_name}/images")
        output_label = os.path.join(path, f"{output_name}_{layer_name}/masks")
        for output_dir in [output_image, output_label]:
            os.makedirs(output_dir, exist_ok=True)
        
        return output_image, output_label

if __name__ == "__main__":
    date = '0205'
    disease = 'PCV'
    PATH = "../../Data/"
    FILE = disease + "_"+ date
    PATH_BASE =  PATH + "/" + disease + "_"+ date
    preprocess = PreprocessData(PATH_BASE)
    # \Data\PCV_0819\trainset\otsu_CC\train\images
    # preprocess.filter_preprocess(FILE + '_otsu',FILE + '_otsu_bil')
    # preprocess.clahe_preprocess(FILE + '_otsu_bil',FILE + '_otsu_bil_clahe')
    # preprocess.preprocess('' ,FILE + '_bil510_clahe7',parm = { "3": [5,10,10,0.7,3],"4":[5,10,10,0.7,12]},bilateralFilter =True ,Unsharp = True,clahe =True,   save_img =True,new_parameter = False,path = './record/')
    preprocess.preprocess(FILE +'' ,FILE + '_fastNlMeansDenoising7_Unsharp_clahe7',parm = { "3": [5,10,10,0.7,3],"4":[5,10,10,0.7,12]},fastNlMeansDenoising=True,bilateralFilter =False ,Unsharp = True,clahe =True,   save_img =True,new_parameter = False,path = './record/')