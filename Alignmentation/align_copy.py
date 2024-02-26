
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import json
import tools.tools as tools
import math
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
# from skimage.metrics import normalized_mutual_info_score
from pathlib import Path
import shutil
import LK_eval
import feature_eval
def setFolder(path):
    os.makedirs(path, exist_ok=True)
def mean_squared_error_ignore_zeros(img1, img2,img3):
    # 對於兩張影像 計算每個pixel 的差異
    # 找出兩影像中相應像素都為零的位置
    both_zeros_mask  = (img1 == 0) & (img2 == 0) & (img3 == 0)

    
    diff = img1[~both_zeros_mask] - img2[~both_zeros_mask]
    mse = np.mean(diff ** 2)
    return mse
def psnr_ignore_zeros(img1, img2,img3):
    
    both_zeros_mask  = (img1 == 0) & (img2 == 0) & (img3 == 0)
    diff = img1[~both_zeros_mask] - img2[~both_zeros_mask] 
    mse = np.mean(diff**2)
    # 如果 MSE 為 0，則 PSNR 為無窮大
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    # 計算 PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def ssim_ignore_zeros(img1, img2,img3):
    both_zeros_mask  = (img1 == 0) & (img2 == 0) & (img3 == 0)
    # 提取非零区域
    img1_non_zero = img1[~both_zeros_mask]
    img2_non_zero = img2[~both_zeros_mask]
    
    if len(img1_non_zero) < 50 or len(img2_non_zero) < 50:
        ssim_index = float('inf')
    else:
        # SSIM
        ssim_index, _ = structural_similarity(img1_non_zero, img2_non_zero, full=True)
        if ssim_index < 0:
            ssim_index = float('inf')
    return ssim_index

def update_metrics_dict(metrics_dict, mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim):
    metrics_dict.update({
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'matching_mse': matching_mse,
        'matching_psnr': matching_psnr,
        'matching_ssim': matching_ssim,
        'relative_change_mse': relative_change_mse,
        'relative_change_psnr': relative_change_psnr,
        'relative_change_ssim': relative_change_ssim
    }) 

def calculate_average_and_std(values):
    avg = round(sum(values) / len(values), 2)
    std = round(np.std(values, ddof=1), 2)
    return [avg, std]

class Matching():
    # Matching(,output_image_path,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL)
    def __init__(self,base_path,image_path,label_path,output_image_path,output_label_path,layers = {"3":"OR","4":"CC"}):
        self.base_path = base_path
        self.image_path = image_path
        self.label_path = label_path
        self.output_image_path = output_image_path
        self.output_label_path = output_label_path
        self.image_size = (304, 304)
        self.data_list = ['1','2', '3', '4']
        self.label_list = ['CC', 'OR']
        self.eval_list = ['mse','psnr','ssim','matching_mse','matching_psnr','matching_ssim']
        self.layers = layers
        
        # template matching
        self.template_matching = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR , cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
        self.template_matching_name = ['TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCORR' , 'TM_CCORR_NORMED', 'TM_CCOEFF', 'TM_CCOEFF_NORMED']
        
        # feature matching
        self.features_matching = ['SIFT','KAZE','AKAZE','ORB','BRISK']
        self.features_matching_matchers = ['BF','FLANN']# 
        self.features_matching_distances = [0.8]# 
        setFolder('./record/'+ disease + '_' + date + '/')
        
        # phase matching
        self.phase_matching = 'PHASE_CORRELATE'
        
        # for data in self.data_list: 
        #     for template_matching in self.template_matching_name:
        #         setFolder( os.path.join(self.output_image_path,template_matching,data))
        #         setFolder( os.path.join(self.output_label_path,template_matching,data))
                
        #     for features_matching in self.features_matching:
        #         for features_matching_matchers in self.features_matching_matchers : 
        #             for features_matching_distances in self.features_matching_distances:
        #                 features_matching_distances = str(features_matching_distances)
                        
        #                 setFolder( os.path.join(self.output_image_path,features_matching+ '_'+features_matching_matchers+'_'+features_matching_distances ,data))
        #                 setFolder( os.path.join(self.output_label_path,features_matching+ '_'+features_matching_matchers+'_'+features_matching_distances ,data))
                
        #     setFolder( os.path.join(self.output_image_path,self.phase_matching,data))
        #     setFolder( os.path.join(self.output_label_path,self.phase_matching,data))
            
    def evaluates(self, pre_treatment_img, post_treatment_img,matching_img):

        cmp_pre_treatment_img = cv2.imread(pre_treatment_img , cv2.IMREAD_GRAYSCALE)
        cmp_post_treatment_img = cv2.imread(post_treatment_img, cv2.IMREAD_GRAYSCALE)
        cmp_matching_img = cv2.imread(matching_img, cv2.IMREAD_GRAYSCALE)
        
        cmp_pre_treatment_img = cv2.resize(cmp_pre_treatment_img,self.image_size)
        cmp_post_treatment_img = cv2.resize(cmp_post_treatment_img,self.image_size)
        cmp_matching_img = cv2.resize(cmp_matching_img,self.image_size)
        
        cmp_pre_treatment_img =cv2.normalize(cmp_pre_treatment_img, None, 0, 255, cv2.NORM_MINMAX)
        cmp_post_treatment_img = cv2.normalize(cmp_post_treatment_img, None, 0, 255, cv2.NORM_MINMAX)
        cmp_matching_img= cv2.normalize(cmp_matching_img, None, 0, 255, cv2.NORM_MINMAX)
        
        # 僅保留 cmp_matching_img 不為0的部分 進行比較
        cmp_pre_treatment_img[cmp_matching_img == 0] = 0
        cmp_post_treatment_img[cmp_matching_img == 0] = 0
        
        # fig , ax = plt.subplots(2,3,figsize=(15,15))
        # ax[0][0].imshow(pre_treatment,cmap='gray')
        # ax[0][0].set_title('pre_treatment')
        # ax[0][0].axis('off')
        # ax[0][1].imshow(post_treatment,cmap='gray')
        # ax[0][1].set_title('post_treatment')
        # ax[0][1].axis('off')
        # ax[0][2].imshow(cmp_matching_img,cmap='gray')
        # ax[0][2].set_title('matching_img')
        # ax[0][2].axis('off')
        # ax[1][0].imshow(cmp_pre_treatment_img,cmap='gray')
        # ax[1][0].set_title('cmp_pre_treatment_img')
        # ax[1][0].axis('off')
        # ax[1][1].imshow(cmp_post_treatment_img,cmap='gray')
        # ax[1][1].set_title('cmp_post_treatment_img')
        # ax[1][1].axis('off')
        # ax[1][2].imshow(cmp_matching_img,cmap='gray')
        # ax[1][2].set_title('cmp_matching_img')
        # ax[1][2].axis('off')
        # plt.show()
        
        
                        
        if cmp_pre_treatment_img is None or cmp_post_treatment_img is None or cmp_matching_img is None:
            return -1 , -1 , -1,-1 , -1 , -1

        if cmp_pre_treatment_img.shape != cmp_post_treatment_img.shape and cmp_post_treatment_img.shape != cmp_matching_img.shape : 
            return -1 , -1 , -1,-1 , -1 , -1

        mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        psnr = psnr_ignore_zeros( cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        ssim = ssim_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        ssim = ssim * 100

        matching_mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_psnr = psnr_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_ssim = ssim_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img ,cmp_matching_img)
        matching_ssim = matching_ssim * 100
        
        # print(mse,psnr,ssim)
        # print(matching_mse,matching_psnr,matching_ssim)

        return mse,psnr,ssim ,matching_mse,matching_psnr,matching_ssim

    def process_matching(self,match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, template_name):
        matching_all[template_name] = {}

        metrics_dict = matching_all[template_name]
        mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim = self.alignment_evaluate(match_path, patient_id, eye, pre_treatment, post_treatment)
        update_metrics_dict(metrics_dict, mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim)
 
    def crop_process_matching(self,match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, template_name):
        matching_all[template_name] = {}

        metrics_dict = matching_all[template_name]
        mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim = self.alignment_evaluate(match_path, patient_id, eye, pre_treatment, post_treatment)
        update_metrics_dict(metrics_dict, mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim)
       
    def process_features_matching(self,match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, features_name):
        matching_all[features_name] = {}
        metrics_dict = matching_all[features_name]
        mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim = self.alignment_evaluate(match_path, patient_id, eye, pre_treatment, post_treatment)
        update_metrics_dict(metrics_dict, mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim)

    def crop_process_features_matching(self,match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, features_name):
        matching_all[features_name] = {}
        metrics_dict = matching_all[features_name]
        mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim = self.alignment_evaluate(match_path, patient_id, eye, pre_treatment, post_treatment)
        update_metrics_dict(metrics_dict, mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim)


    def process_phase_matching(self,match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, phase_matching):
        matching_all[phase_matching] = {}
        metrics_dict = matching_all[phase_matching]
        mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim = self.alignment_evaluate(match_path, patient_id, eye, pre_treatment, post_treatment)
        update_metrics_dict(metrics_dict, mse, psnr, ssim, matching_mse, matching_psnr, matching_ssim, relative_change_mse, relative_change_psnr, relative_change_ssim)

    def find_best_method(self, matching_all):
        difference_ssim = -1
        difference_psnr = -1
        difference_mse = -1
        best_method = None
        
        for method in matching_all:
            matching_ssim = matching_all[method]['matching_ssim']
            ssim = matching_all[method]['ssim']
            matching_psnr = matching_all[method]['matching_psnr']
            psnr = matching_all[method]['psnr']
            matching_mse = matching_all[method]['matching_mse']
            mse = matching_all[method]['mse']
            
            diff_ssim, diff_psnr,diff_mse = matching_ssim - ssim, matching_psnr - psnr,matching_mse-mse
            
            if diff_ssim > difference_ssim:
                difference_ssim = diff_ssim
                difference_psnr = diff_psnr
                difference_mse = diff_mse
                best_method= method
            elif diff_psnr > difference_psnr:
                difference_ssim = diff_ssim
                difference_psnr = diff_psnr
                difference_mse = diff_mse
            elif diff_mse < difference_mse :
                difference_ssim = diff_ssim
                difference_psnr = diff_psnr
                difference_mse = diff_mse  
                
        if  best_method is None:
            best_method =  method            
                     
        return best_method            
                            
            
    def feature_generate(self) :
        feature = LK_eval.LK(self.label_path,self.image_path,self.output_label_path,self.output_image_path,self.features_matching,self.features_matching_matchers,self.features_matching_distances)   
        feature_crop = feature_eval.finding(self.base_path,self.label_path,self.image_path,self.output_label_path,self.output_image_path,self.features_matching,self.features_matching_matchers,self.features_matching_distances)
        for features_matching in self.features_matching:
            for features_matching_matchers in self.features_matching_matchers:
                for features_matching_distances in self.features_matching_distances:
                    feature_dict = feature.registration(features_matching,features_matching_matchers,features_matching_distances)
                    feature_crop_dict = feature_crop.feature(features_matching,features_matching_matchers,features_matching_distances)
                    

                    
    
    def final_feature(self):

        patient = {}
        best_case = {}
        worst_case = {}
        best_diff_ssim, best_diff_psnr = 0,0
        worst_diff_ssim, worst_diff_psnr = 100000000000000000000, 100000000000000000000
        mse_list, psnr_list, ssim_list = [], [], []
        matching_mse_list, matching_psnr_list, matching_ssim_list = [], [], []
        

            
        image_folder = self.image_path+ '1/'
        filenames = sorted(os.listdir(image_folder))
        best_method_count = {}
                    
        for filename in filenames: 
            if filename.endswith(".png"):
                patient_id, eye, date = filename.split('.png')[0].split('_')
                if patient_id + '_' + eye not in patient :
                    patient[patient_id + '_' + eye] = {'pre_date': date}
                    pre_treatment = date
                    post_treatment = ''

                else :
                    post_treatment =  date 
                    if 'post_date' not in patient[patient_id + '_' + eye] :
                        patient[patient_id + '_' + eye]['post_date'] = {}
                    
                    patient[patient_id + '_' + eye]['post_date'][post_treatment]={}
                    matching_all = {}
                    if 'pre_date' in patient[patient_id + '_' + eye] and post_treatment in patient[patient_id + '_' + eye]['post_date']:
                        for template_matching in self.template_matching_name:
                            # match_path = os.path.join(self.output_image_path,template_matching)
                            # self.process_matching(match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, template_matching)
                            features_name = f"crop_{template_matching}"
                            match_path = os.path.join(self.output_image_path,features_name)
                            self.process_matching(match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, features_name)
                        # for template_matching in self.template_matching_name:
                        #     features_name = f"crop_{template_matching}"
                        #     match_path = os.path.join(self.output_image_path,features_name)
                        #     self.crop_process_matching(match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, features_name)
                            
                        for features_matching in self.features_matching:
                            for features_matching_matchers in self.features_matching_matchers:
                                for features_matching_distances in self.features_matching_distances:
                                    # features_name = f"{features_matching}_{features_matching_matchers}_{features_matching_distances}"
                                    # match_path = os.path.join(self.output_image_path, features_name)
                                    # self.process_features_matching(match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, features_name)
                                    features_name = f"crop_{features_matching}_{features_matching_matchers}_{features_matching_distances}"
                                    match_path = os.path.join(self.output_image_path, features_name)
                                    self.process_features_matching(match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, features_name)
                                    
                        # for features_matching in self.features_matching:
                        #     for features_matching_matchers in self.features_matching_matchers:
                        #         for features_matching_distances in self.features_matching_distances:
                        #             features_name = f"crop_{features_matching}_{features_matching_matchers}_{features_matching_distances}"
                        #             match_path = os.path.join(self.output_image_path, features_name)
                        #             self.process_features_matching(match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, features_name)


                        # match_path = os.path.join(self.output_image_path, self.phase_matching)
                        # self.process_phase_matching(match_path, matching_all, patient_id, eye, pre_treatment, post_treatment, self.phase_matching)
                    
                        
                        best_method = self.find_best_method(matching_all)
                        
                        for data in self.data_list :
                            setFolder(os.path.join(self.base_path, 'align','ALL', data))
                        for layer in  self.layers.values():
                            setFolder(os.path.join(self.base_path, 'align',layer,'images'))
                            setFolder(os.path.join(self.base_path, 'align',layer,'masks'))
                            
                        for data in self.data_list :
                            treatment_img = os.path.join(self.output_image_path, best_method) + '/' + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                            shutil.copy(treatment_img , self.base_path + '/' + 'align' + '/'  + 'ALL' + '/'+ data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                            if data == '3' or data == '4' : 
                                shutil.copy(treatment_img , self.base_path + '/' + 'align' + '/'  + self.layers[data] + '/'  + 'images'+'/'+ patient_id + '_' + eye + '_' + pre_treatment + '.png')
                                # shutil.copy(treatment_label , self.base_path + '/' + 'align' + '/' + label + '/' + 'masks'+'/'+ patient_id + '_' + eye + '_' + pre_treatment + '.png')
                            
                        for label in self.label_list:
                            
                            treatment_label = os.path.join(self.output_label_path, best_method) + '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                            if os.path.exists(treatment_label) :
                                shutil.copy(treatment_label , self.base_path + '/' + 'align' + '/' + label + '/' + 'masks'+'/'+ patient_id + '_' + eye + '_' + pre_treatment + '.png')
                        
                        
                        if best_method  not in best_method_count:
                            best_method_count[best_method] = 1
                        else:
                            best_method_count[best_method] += 1
                            


                                   
                        for data in self.data_list :
                            treatment_img = os.path.join(self.output_image_path, best_method) + '/' + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                            shutil.copy(treatment_img , self.base_path + '/' + 'align' + '/'  + 'ALL' + '/'+ data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                            if data == '3' or data == '4' : 
                                shutil.copy(treatment_img , self.base_path + '/' + 'align' + '/'  + self.layers[data] + '/'  + 'images'+'/'+ patient_id + '_' + eye + '_' + post_treatment + '.png')
                                
                            
                        for label in self.label_list:
                            
                            treatment_label = os.path.join(self.output_label_path, best_method) + '/' + label + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                            if os.path.exists(treatment_label) :
                                shutil.copy(treatment_label , self.base_path + '/' + 'align' + '/' + label + '/' + 'masks'+'/'+ patient_id + '_' + eye + '_' + post_treatment + '.png')
                           
                        mse = matching_all[best_method]['mse']
                        psnr = matching_all[best_method]['psnr']
                        ssim = matching_all[best_method]['ssim']
                        matching_mse = matching_all[best_method]['matching_mse']
                        matching_psnr = matching_all[best_method]['matching_psnr']
                        matching_ssim = matching_all[best_method]['matching_ssim']
                        patient[patient_id + '_' + eye]['post_date'][date]['best_method'] = {}
                        best_method_case = patient[patient_id + '_' + eye]['post_date'][date]['best_method']
                        best_method_case.update({
                            'best_method': best_method, 
                            'psnr': psnr, 
                            'ssim': ssim, 
                            'mse': mse,
                            'matching_psnr': matching_psnr, 
                            'matching_ssim': matching_ssim, 
                            'matching_mse': matching_mse
                        })
                        
                        mse_list.append(mse)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        matching_mse_list.append(matching_mse)
                        matching_psnr_list.append(matching_psnr)
                        matching_ssim_list.append(matching_ssim) 
                                             
                        
                        diff_ssim, diff_psnr = matching_ssim - ssim, matching_psnr - psnr
                            
                        if diff_ssim > best_diff_ssim:
                            best_diff_ssim = diff_ssim
                            best_case.update({
                                'patient': [patient_id, eye, post_treatment], 
                                'psnr': psnr, 
                                'ssim': ssim, 
                                'mse': mse,
                                'matching_psnr': matching_psnr, 
                                'matching_ssim': matching_ssim, 
                                'matching_mse': matching_mse
                                })

                        elif diff_psnr > best_diff_psnr:
                            best_diff_psnr = diff_psnr
                            best_case.update({
                                'patient': [patient_id, eye, post_treatment], 
                                'psnr': psnr, 
                                'ssim': ssim, 
                                'mse': mse,
                                'matching_psnr': matching_psnr, 
                                'matching_ssim': matching_ssim, 
                                'matching_mse': matching_mse
                                })

                        if diff_ssim < worst_diff_ssim:
                            worst_diff_ssim = diff_ssim
                            worst_case.update({
                                'patient': [patient_id, eye, post_treatment], 
                                'psnr': psnr, 
                                'ssim': ssim, 
                                'mse': mse,
                                'matching_psnr': matching_psnr, 
                                'matching_ssim': matching_ssim, 
                                'matching_mse': matching_mse
                                })

                        elif diff_psnr < worst_diff_psnr:
                            worst_diff_psnr = diff_psnr
                            worst_case.update({
                                'patient': [patient_id, eye, post_treatment], 
                                'psnr': psnr, 
                                'ssim': ssim, 
                                'mse': mse,
                                'matching_psnr': matching_psnr, 
                                'matching_ssim': matching_ssim, 
                                'matching_mse': matching_mse
                                })
                            
        matching_avg_mse = round(sum(matching_mse_list)/len(matching_mse_list),2)
        matching_avg_psnr = round(sum(matching_psnr_list)/len(matching_psnr_list),2)
        matching_avg_ssim = round(sum(matching_ssim_list)/len(matching_ssim_list),2)
        matching_mse_std = round(np.std(matching_mse_list, ddof=1),2)
        matching_psnr_std = round(np.std(matching_psnr_list, ddof=1),2)
        matching_ssim_std = round(np.std(matching_ssim_list, ddof=1),2)

        patient['avg'] = {
            'mse':matching_avg_mse,
            'mse_std':matching_mse_std,
            'psnr':matching_avg_psnr,
            'psnr_std':matching_psnr_std,
            'ssim':matching_avg_ssim,
            'ssim_std':matching_avg_ssim,
        }


        if best_case :
            patient['best_case'] = best_case

        if worst_case :
            patient['worst_case'] = worst_case

        if mse_list:    
            avg_mse = round(sum(mse_list)/len(mse_list),2)
            avg_psnr = round(sum(psnr_list)/len(psnr_list),2)
            avg_ssim = round(sum(ssim_list)/len(ssim_list),2)
            mse_std = round(np.std(mse_list, ddof=1),2)
            psnr_std = round(np.std(psnr_list, ddof=1),2)
            ssim_std = round(np.std(ssim_list, ddof=1),2)

            patient['original'] = {
                'mse':matching_avg_mse,
                'mse_std':matching_mse_std,
                'psnr':matching_avg_psnr,
                'psnr_std':matching_psnr_std,
                'ssim':matching_avg_ssim,
                'ssim_std':matching_avg_ssim,
            }
       
        return patient ,best_method_count                        

                        
                    
    def alignment_evaluate(self,match_path,patient_id,eye,pre_treatment,post_treatment):
        pre_treatment_img = self.image_path+ '/1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
        post_treatment_img = self.image_path + '/1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
        matching_img = match_path  + '/1_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
        mse,psnr,ssim, matching_mse,matching_psnr,matching_ssim = self.evaluates(pre_treatment_img,post_treatment_img,matching_img)
        if psnr == float('inf') :
            return -1 , -1 , -1,-1 , -1 , -1,-1 , -1 , -1 
        if ssim == float('inf') : 
            return -1 , -1 , -1,-1 , -1 , -1,-1 , -1 , -1 
        if matching_psnr == float('inf') :
            return -1 , -1 , -1,-1 , -1 , -1,-1 , -1 , -1 
        if matching_ssim == float('inf')  :
            return -1 , -1 , -1,-1 , -1 , -1,-1 , -1 , -1 
        if ssim  < 0 or psnr < 0 or mse <0 or matching_ssim  < 0 or  matching_psnr < 0 or matching_mse <0 :
            return -1 , -1 , -1,-1 , -1 , -1,-1 , -1 , -1 

        relative_change_ssim = (matching_ssim - ssim)
        relative_change_psnr  = (matching_psnr - psnr)
        relative_change_mse = (mse - matching_mse)
        
        
        return mse,psnr,ssim, matching_mse,matching_psnr,matching_ssim ,relative_change_mse,relative_change_psnr,relative_change_ssim 
        
                        

  
            
 
    # def run(self,):
    #     for distance in distances:
    #         for feature in features:
    #             for matcher in matchers:
                    

                
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
    
    
    
    
    matching = Matching(PATH_BASE,image_path,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL)
    # matching.feature_generate()
    patient,best_method_count = matching.final_feature()
    json_file = './record/'+ disease + '_' + date + '/'+ 'align.json'
    tools.write_to_json_file(json_file, patient)
    print(best_method_count)
    json_file = './record/'+ disease + '_' + date + '/'+ 'best_method_count.json'
    tools.write_to_json_file(json_file, best_method_count)
    
                