import cv2
import os
import numpy as np
import pathlib as pl
import shutil
import pandas as pd
import csv
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

# Entropy

import tools.tools as tools
class template_matcher():

        
    def __init__(self,image_path,label_path,output_image_path,output_label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.output_image_path = output_image_path
        self.output_label_path = output_label_path
        self.data_list = ['1','2', '3', '4', '1_OCT', '2_OCT', '3_OCT', '4_OCT']
        self.label_list = ['CC', 'OR']
        self.image_size = (304, 304)
        
        setFolder(self.output_image_path)
        setFolder(self.output_label_path)

        self.methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR , cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
        self.method_name = ['TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCORR' , 'TM_CCORR_NORMED', 'TM_CCOEFF', 'TM_CCOEFF_NORMED']
        for method in self.method_name:
            for data in self.data_list:
                setFolder(os.path.join(self.output_image_path, method, data))
            for label in self.label_list:
                setFolder(os.path.join(self.output_label_path, method, label))


    #Template Match Method:找到template在img中最高的相關係數的數值(r)和位置(top_left)
    def getPoints(self,img, template, method=cv2.TM_CCOEFF_NORMED):
        result = cv2.matchTemplate(img, template, method) # 回傳的是相關係數矩陣
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # 找到最大值和最小值的位置 也就是左上角的位置 以及右下角的位置
        if self.method_name[method] in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
            top_left = min_loc # 要移動的位置
            r = min_val # r=最低的相關係數
        else:
            top_left = max_loc
            r = max_val # r=最高的相關係數
        return top_left, r #top_left:回傳對位左上角的位置, r=最高的相關係數


    def get_element(self,image, template, offset_x=0, offset_y=0, method=cv2.TM_CCOEFF_NORMED):
        elements, r = self.getPoints(image, template, method)
        elements = (elements[0] - offset_x, elements[1] - offset_y)
        return elements, r

    #將術後的影像切成4個角落跟中央，並分別跟術前影像做getPoints，找到5個template中R最高的位置，並回傳要位移的x跟y
    def pointMatch(self,image, template, method = cv2.TM_CCOEFF_NORMED):
        # template_total      = template
        template_center     = template[52:52+199, 52:52+199]
        template_left_up    = template[0:199, 0:199]
        template_right_up   = template[104:104+199, 0:199]
        template_left_down  = template[0:199, 104:104+199]
        template_right_down = template[104:104+199, 104:104+199]

        # elements_total, r_t = self.get_element(image, template_total,0,0,method)
        elements_c, r_c = self.get_element(image, template_center, 52, 52, method)
        elements_TL, r_1 = self.get_element(image, template_left_up, 0, 0, method)
        elements_TR, r_2 = self.get_element(image, template_right_up, 0, 104, method)
        elements_DL, r_3 = self.get_element(image, template_left_down, 104, 0, method)
        elements_DR, r_4 = self.get_element(image, template_right_down, 104, 104, method)
        #----------------------------------------可視化用----------------------------------------
        # show template
        # fig ,ax = plt.subplots(2,3) 
        # plt rgb 
        # template_total = cv2.cvtColor(template_total,cv2.COLOR_GRAY2BGR)
        # template_center = cv2.cvtColor(template_center,cv2.COLOR_GRAY2BGR)
        # template_left_up = cv2.cvtColor(template_left_up,cv2.COLOR_GRAY2BGR)
        # template_right_up = cv2.cvtColor(template_right_up,cv2.COLOR_GRAY2BGR)
        # template_left_down = cv2.cvtColor(template_left_down,cv2.COLOR_GRAY2BGR)
        # template_right_down = cv2.cvtColor(template_right_down,cv2.COLOR_GRAY2BGR)


        # ax[0,0].imshow(template_total)
        # ax[0,0].set_title('template_total')
        # ax[0,0].axis('off' )

        # ax[0,1].imshow(template_center, cmap='gray') 
        # ax[0,1].set_title('template_center')
        # ax[0,1].axis('off')

        # ax[0,2].imshow(template_left_up)
        # ax[0,2].set_title('template_left_up')
        # ax[0,2].axis('off')

        # ax[1,0].imshow(template_right_up)
        # ax[1,0].set_title('template_right_up')
        # ax[1,0].axis('off')

        # ax[1,1].imshow(template_left_down)
        # ax[1,1].set_title('template_left_down')
        # ax[1,1].axis('off')
        # ax[1,2].imshow(template_right_down)
        # ax[1,2].set_title('template_right_down')
        # ax[1,2].axis('off')

        # plt.show()
        # plt.pause(0.1)

        # print('Total : ', round(r_t,3) , " shift(x,y) :" + str(elements_total))
        # print('Center : ', round(r_c,3), " shift(x,y) :" + str(elements_c))
        # print('TL : ', round(r_1,3), " shift(x,y) :" + str(elements_TL))
        # print('TR : ', round(r_2,3), " shift(x,y) :" + str(elements_TR))
        # print('DL : ', round(r_3,3), " shift(x,y) :" + str(elements_DL))
        # print('DR : ', round(r_4,3), " shift(x,y) :" + str(elements_DR))
        #----------------------------------------可視化用----------------------------------------

        # 選擇最高的相關係數的位置
        e = [elements_c, elements_TL, elements_TR, elements_DL, elements_DR]
        r = [r_c, r_1, r_2, r_3, r_4]
        Relation = 0
        relation_min = 1000000000000000000000000000
        shift_x = None
        shift_y = None
        for i in range(0, len(r)):

            if self.method_name[method] in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:

                if r[i] < relation_min:
                    relation_min = r[i]
                    shift_x = e[i][0]
                    shift_y = e[i][1]
            elif  self.method_name[method] in ['TM_CCORR']:  
                if r[i] > Relation:
                    Relation = r[i]
                    shift_x = e[i][0]
                    shift_y = e[i][1]
            else:
                if r[i] > Relation:
                    Relation = r[i]
                    shift_x = e[i][0]
                    shift_y = e[i][1]                
        if shift_x != None : 
            return shift_x, shift_y
        else:
            print('No Match')
            return 0,0

    def evaluate(self,image ,original_image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
        mse = mean_squared_error(image, original_image)
        psnr = peak_signal_noise_ratio(image, original_image)
        ssim = structural_similarity(image, original_image)
        ssim = ssim * 100
        # Calculate the mean of the two images
        mean_image1 = np.mean(original_image)
        mean_image2 = np.mean(image)

        # Calculate the NCC
        ncc = np.sum((original_image - mean_image1) * (image - mean_image2)) / (
            np.sqrt(np.sum((original_image - mean_image1)**2) * np.sum((image - mean_image2)**2))
        )

        return mse,psnr,ssim,ncc

    def get_pre_treatment_evaluation(self,file_path):
        patient_dict = read_patient_list(file_path)
        avg_mse = 0
        avg_psnr = 0
        avg_ssim = 0
        count = 0
        mse_list = []
        psnr_list = []
        ssim_list = []
        ncc_list = []
        for patient, eyes in patient_dict.items():
            for eye,date_list in eyes.items():
                if len(date_list) > 1:
                    path_pre_date = pl.Path(self.image_path + patient + '/'+date_list[0] + '/' + eye )
                    if path_pre_date.exists() and path_pre_date.is_dir() :
                        
                        for date in date_list[1:]:
                            path_date = pl.Path(self.image_path + patient + '/'+date + '/' + eye )
                            if path_date.exists() and path_date.is_dir() :
                                img_mse = []
                                img_psnr = []
                                img_ssim = []
                                img_ncc = []
                                img_avg_mse = -1
                                img_avg_psnr = -1
                                img_avg_ssim = -1
                                img_avg_ncc = -1
                                best_case = dict()
                                best_ssim = -1
                                worst_case = dict()
                                worst_ssim = 10000000000000
                                for img in self.data_list:
                                    pre_treatment =  self.image_path + patient + '/'+date_list[0] + '/' + eye + '/' + img + '.png'
                                    original = self.image_path + patient + '/'+date + '/' + eye + '/' + img + '.png'
                                    mse,psnr,ssim,ncc = self.evaluate(pre_treatment,original)
                                    img_mse.append(mse)
                                    img_psnr.append(psnr)
                                    img_ssim.append(ssim)
                                img_avg_mse  = sum(img_mse)/len(img_mse)
                                img_avg_psnr = sum(img_psnr)/len(img_psnr)
                                img_avg_ssim = sum(img_ssim)/len(img_ssim)
                                if img_avg_ssim > best_ssim :
                                    best_ssim = img_avg_ssim
                                    best_case['patient'] = [ patient,eye,date]
                                    best_case ['psnr'] = img_avg_psnr
                                    best_case ['ssim'] = img_avg_ssim
                                    best_case ['mse'] = img_avg_mse
                                    best_case ['ncc'] = img_avg_ncc
                                    
                                if img_avg_ssim < worst_ssim :
                                    worst_ssim = img_avg_ssim
                                    worst_case['patient'] = [ patient,eye,date]
                                    worst_case ['psnr'] = img_avg_psnr
                                    worst_case ['ssim'] = img_avg_ssim
                                    worst_case ['mse'] = img_avg_mse
                                    worst_case ['ncc'] = img_avg_ncc
                                    
                                mse_list.append(img_avg_mse)
                                psnr_list.append(img_avg_psnr)
                                ssim_list.append(img_avg_ssim)
                                ncc_list.append(img_avg_ncc)


        avg_mse = round(sum(mse_list)/len(mse_list),5)
        avg_psnr = round( sum(psnr_list)/len(psnr_list),5)
        avg_ssim = round( sum(ssim_list)/len(ssim_list),5)
        avg_ncc = round( sum(ncc_list)/len(ncc_list),5)
        mse_std = round(np.std(mse_list, ddof=1),5)
        psnr_std = round(np.std(psnr_list, ddof=1),5)
        ssim_std = round(np.std(ssim_list, ddof=1),5)
        ncc_std = round(np.std(ncc_list, ddof=1),5)


        eval = {}
        eval['avg_mse'] = avg_mse
        eval['mse_std'] = mse_std
        eval['avg_psnr'] = avg_psnr
        eval['psnr_std'] = psnr_std
        eval['avg_ssim'] = avg_ssim
        eval['ssim_std'] = ssim_std
        eval['avg_ncc'] = avg_ncc
        eval['ncc_std'] = ncc_std
        eval['best_case'] = best_case
        eval['worst_case'] = worst_case
        return eval

    
    def avg_evaluate(self):
        eval = {}
        original = {}

        for method in pl.Path(self.image_path).iterdir():
            print(method)
            if method.is_dir():
                method = method.name

                eval[method] = {}
                patient = {}
                image_folder = self.image_path+ '/1/'
                filenames = sorted(os.listdir(image_folder))

                best_case = dict()
                worst_case = dict()
                best_differece_ssim = -1
                worst_differece_ssim = 100000000000000000000
                avg_mse = 0
                avg_psnr = 0
                avg_ssim = 0
                mse_list = []
                psnr_list = []
                ssim_list = []
                ncc_list = []
                matching_mse_list = []
                matching_psnr_list = []
                matching_ssim_list = []
                matching_ncc_list = []
                for filename in filenames: 
                    if filename.endswith(".png"):

                        patient_id, eye, date = filename.split('.png')[0].split('_')
                        if patient_id + '_' + eye not in patient :
                            patient[patient_id + '_' + eye] = {}
                            post_treatment = ''
                            pre_treatment = date
                        else :
                            post_treatment =  date 

                            pre_treatment_img = self.image_path+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                            post_treatment_img = self.image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                            matching_img = self.image_path + method + '/1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                            mse,psnr,ssim,ncc = self.evaluate(pre_treatment_img,post_treatment_img)
                            mse_list.append(mse)
                            psnr_list.append(psnr)
                            ssim_list.append(ssim)
                            ncc_list.append(ncc)


                            matching_mse,matching_psnr,matching_ssim,matching_ncc = self.evaluate(pre_treatment_img,matching_img)
                            matching_mse_list.append(matching_mse)
                            matching_psnr_list.append(matching_psnr)
                            matching_ssim_list.append(matching_ssim)
                            matching_ncc_list.append(matching_ncc)

                            if matching_ssim - ssim > best_differece_ssim :
                                best_differece_ssim = matching_ssim - ssim
                                best_case['patient'] = [ patient_id,eye,post_treatment]
                                best_case ['psnr'] = psnr
                                best_case ['ssim'] = ssim
                                best_case ['mse'] = mse
                                best_case ['ncc'] = ncc
                                best_case ['matching_psnr'] = matching_psnr
                                best_case ['matching_ssim'] = matching_ssim
                                best_case ['matching_mse'] = matching_mse
                                best_case ['matching_ncc'] = matching_ncc

                            if matching_ssim - ssim < worst_differece_ssim :
                                worst_differece_ssim = matching_ssim - ssim
                                worst_case['patient'] = [ patient_id,eye,post_treatment]
                                worst_case ['psnr'] = psnr
                                worst_case ['ssim'] = ssim
                                worst_case ['mse'] = mse
                                worst_case ['ncc'] = ncc
                                worst_case ['matching_psnr'] = matching_psnr
                                worst_case ['matching_ssim'] = matching_ssim
                                worst_case ['matching_mse'] = matching_mse
                                worst_case ['matching_ncc'] = matching_ncc


                matching_avg_mse = round(sum(matching_mse_list)/len(matching_mse_list),3)
                matching_avg_psnr = round(sum(matching_psnr_list)/len(matching_psnr_list),3)
                matching_avg_ssim = round(sum(matching_ssim_list)/len(matching_ssim_list),3)
                matching_avg_ncc = round(sum(matching_ncc_list)/len(matching_ncc_list),3)
                matching_mse_std = round(np.std(matching_mse_list, ddof=1),3)
                matching_psnr_std = round(np.std(matching_psnr_list, ddof=1),3)
                matching_ssim_std = round(np.std(matching_ssim_list, ddof=1),3)
                matching_ncc_std = round(np.std(matching_ncc_list, ddof=1),3)
                eval[method]['mse'] = [matching_avg_mse,matching_mse_std]
                eval[method]['psnr'] = [matching_avg_psnr,matching_psnr_std]
                eval[method]['ssim'] = [matching_avg_ssim,matching_ssim_std]
                eval[method]['ncc'] = [matching_avg_ncc,matching_ncc_std]

                if best_case != {} :
                    eval[method]['best_case'] = best_case
                if worst_case != {} :
                    eval[method]['worst_case'] = worst_case




                if original == {} :
                    avg_mse = round(sum(mse_list)/len(mse_list),3)
                    avg_psnr = round(sum(psnr_list)/len(psnr_list),3)
                    avg_ssim = round(sum(ssim_list)/len(ssim_list),3)
                    avg_ncc = round(sum(ncc_list)/len(ncc_list),3)
                    mse_std = round(np.std(mse_list, ddof=1),3)
                    psnr_std = round(np.std(psnr_list, ddof=1),3)
                    ssim_std = round(np.std(ssim_list, ddof=1),3)
                    ncc_std = round(np.std(ncc_list, ddof=1),3)

                    eval['original'] = {}
                    eval['original']['mse'] = [avg_mse,mse_std]
                    eval['original']['psnr'] = [avg_psnr,psnr_std]
                    eval['original']['ssim'] = [avg_ssim,ssim_std]
                    eval['original']['ncc'] = [avg_ncc,ncc_std]



        return eval







            # for patient, eyes in patient_dict.items():
            #     for eye,date_list in eyes.items():
            #         if len(date_list) > 1:
            #             path_pre_date = pl.Path(self.image_path + method + '/' +patient + '/'+date_list[0] + '/' + eye )
            #             if path_pre_date.exists() and path_pre_date.is_dir() :
            #                 for date in date_list[1:]:
            #                     path_date = pl.Path(self.image_path + method + '/' +patient + '/'+date + '/' + eye )
            #                     if path_date.exists() and path_date.is_dir() :
            #                         img_avg_mse = -1
            #                         img_avg_psnr = -1
            #                         img_avg_ssim = -1
            #                         img_avg_ncc = -1
            #                         img_avg_o_mse = -1
            #                         img_avg_o_psnr = -1
            #                         img_avg_o_ssim = -1
            #                         img_avg_o_ncc = -1
            #                         img_mse = []
            #                         img_psnr = []
            #                         img_ssim = []
            #                         img_ncc = []
            #                         img_o_mse = []
            #                         img_o_psnr = []
            #                         img_o_ssim = []
            #                         img_o_ncc = []
            #                         for img in self.data_list:

            #                             pre_treatment_original =  self.image_path + patient + '/'+date_list[0] + '/' + eye + '/'  + img + '.png'
            #                             pre_treatment =  self.image_path + method + '/' +patient + '/'+date_list[0] + '/' + eye + '/' + img + '.png'

            #                             post_tratment_original = self.image_path + patient + '/'+date + '/' + eye + '/' + img + '.png'
            #                             post_tratment = self.image_path + method + '/' +patient + '/'+date + '/' + eye + '/' +img + '.png'

            #                             mse,psnr,ssim,ncc = self.evaluate(pre_treatment,post_tratment)

            #                             mse_o ,psnr_o,ssim_o,ncc = self.evaluate(pre_treatment_original,post_tratment_original)
                                        

            #                             img_mse.append(mse)
            #                             img_psnr.append(psnr)
            #                             img_ssim.append(ssim)
            #                             img_ncc.append(ncc)
            #                             img_o_mse.append(mse_o)
            #                             img_o_psnr.append(psnr_o)
            #                             img_o_ssim.append(ssim_o)
            #                             img_o_ncc.append(ncc)

            #                         img_avg_mse = sum(img_mse)/len(img_mse)
            #                         img_avg_psnr = sum(img_psnr)/len(img_psnr)
            #                         img_avg_ssim = sum(img_ssim)/len(img_ssim)
            #                         img_avg_ncc = sum(img_ncc)/len(img_ncc)
            #                         img_avg_o_mse = sum(img_o_mse)/len(img_o_mse)
            #                         img_avg_o_psnr = sum(img_o_psnr)/len(img_o_psnr)
            #                         img_avg_o_ssim = sum(img_o_ssim)/len(img_o_ssim)
            #                         img_avg_o_ncc = sum(img_o_ncc)/len(img_o_ncc)
            #                         # print('img_avg_mse',sum(img_mse),len(img_mse),img_mse)
            #                         # print('img_avg_o_mse',sum(img_o_mse),len(img_o_mse),img_o_mse)
            #                         # print('img_avg_psnr',sum(img_psnr),len(img_psnr),  img_psnr)
            #                         # print('img_avg_o_psnr',sum(img_o_psnr),len(img_o_psnr),img_o_psnr)

            #                         # print('img_avg_ssim',sum(img_ssim),len(img_ssim),img_ssim)
            #                         # print('img_avg_o_ssim',sum(img_o_ssim),len(img_o_ssim),img_o_ssim)
            #                         # print('img_avg_psnr - img_avg_o_psnr',img_avg_psnr - img_avg_o_psnr)
            #                         mse_list.append(img_avg_mse)
            #                         psnr_list.append(img_avg_psnr)
            #                         ssim_list.append(img_avg_ssim)
            #                         ncc_list.append(img_avg_ncc)
            #                         # print('mse_list',mse_list)
            #                         # print('psnr_list',psnr_list)
            #                         # print('ssim_list',ssim_list)
            #                         # print('best_differece_ssim',best_differece_ssim)
            #                         if img_avg_ssim - img_avg_o_ssim > best_differece_ssim:
            #                             best_differece_ssim = img_avg_psnr - img_avg_o_psnr
            #                             best_case['mse']= [img_avg_o_mse , img_avg_mse]
            #                             best_case['psnr']= [img_avg_o_psnr , img_avg_psnr]
            #                             best_case['ssim']= [img_avg_o_ssim , img_avg_ssim]
            #                             best_case['ncc'] = [img_avg_o_ncc , img_avg_ncc]
            #                             best_case['patient']= [patient,eye,date]

            #                         if img_avg_ssim - img_avg_o_ssim < worst_differece_ssim:
            #                             worst_differece_ssim = img_avg_psnr - img_avg_o_psnr
            #                             worst_case['mse']= [img_avg_o_mse , img_avg_mse]
            #                             worst_case['psnr']= [img_avg_o_psnr , img_avg_psnr]
            #                             worst_case['ssim']= [img_avg_o_ssim , img_avg_ssim]
            #                             worst_case['ncc'] = [img_avg_o_ncc , img_avg_ncc]
            #                             worst_case['patient']= [patient,eye,date]

            # avg_mse = round(sum(mse_list)/len(mse_list),5)
            # avg_psnr = round(sum(psnr_list)/len(psnr_list),5)
            # avg_ssim = round(sum(ssim_list)/len(ssim_list),5)
            # avg_ncc = round(sum(ncc_list)/len(ncc_list),5)
            # mse_std = round(np.std(mse_list, ddof=1),5)
            # psnr_std = round(np.std(psnr_list, ddof=1),5)
            # ssim_std = round(np.std(ssim_list, ddof=1),5)
            # ncc_std = round(np.std(ncc_list, ddof=1),5)
            # eval[method]['mse'] = avg_mse
            # eval[method]['psnr'] = avg_psnr
            # eval[method]['ssim'] = avg_ssim
            # eval[method]['ncc'] = avg_ncc
            # eval[method]['best_case'] = best_case
            # eval[method] ['worst_case'] = worst_case
            # eval[method] ['mse_std'] = mse_std
            # eval[method] ['psnr_std'] = psnr_std
            # eval[method] ['ssim_std'] = ssim_std
            # eval[method] ['ncc_std'] = ncc_std

        return eval 


    # # template matching
    # def template_matching(self,pre_img_path, post_img_path):
    #     print(self.image_path +pre_img_path)
    #     if not os.path.exists(self.image_path +pre_img_path) or not os.path.exists(self.image_path +post_img_path):
    #         print('pre_img_path not exist')
    #         return
    #     if not os.path.exists(self.image_path + pre_img_path + '/1.png') or not os.path.exists(self.image_path +post_img_path + '/1.png'):
    #         print('pre_img_path not exist')
    #         return
    #     image_pre = cv2.imread(self.image_path + pre_img_path + '/1.png')
    #     gray_pre = cv2.cvtColor(image_pre, cv2.COLOR_BGR2GRAY)
    #     image_post = cv2.imread(self.image_path +post_img_path + '/1.png')
    #     gray_post = cv2.cvtColor(image_post, cv2.COLOR_BGR2GRAY)
        


    #     color = [(0,1),(0,1),(1,2),(1,2),(2,0),(2,0)]
    #     shift = dict()
    #     for method in self.methods:
    #         match_par = self.method_name[self.methods.index(method)]
    #         # print method name
    #         shift[match_par]= []
    #         shift_x, shift_y = self.pointMatch(gray_pre, gray_post, method)
    #         shift[match_par] = [shift_x, shift_y]
    #         # 進行平移變換
    #         M = np.float32([[1, 0, shift_x], [0, 1, shift_y]]) # 平移矩陣

    #         for data in self.data_list:
    #             if not os.path.exists(self.image_path +post_img_path + '/' + data + '.png'):
    #                 print('post_img_path not exist')
    #                 return
    #             if  not os.path.exists(self.image_path + post_img_path + data + '.png' ):
    #                 print('post_img_path not exist')
    #                 return
    #             image = cv2.imread(self.image_path + post_img_path + data + '.png')
    #             result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    #             result[result == 0] = image [result == 0]
    #             if not os.path.exists(self.image_path  + match_par+ '/' + post_img_path):
    #                 os.makedirs(self.image_path  + match_par+ '/' + post_img_path)
    #             cv2.imwrite(self.image_path  + match_par+ '/' + post_img_path + data + '.png', result)


    #             vis_img = image.copy()
    #             vis_img[:,:,0] = 0
    #             vis_img[:,:,2] = 0

    #             image_pre = cv2.imread(self.image_path + pre_img_path + data + '.png')
    #             add = cv2.addWeighted(image_pre, 0.5, vis_img, 1.0, 0)


    #             # cv2.imwrite(self.image_path  + match_par+ '/' + post_img_path + data + '_vis.png', vis_img)
    #             # cv2.imwrite(self.image_path  + match_par+ '/' + post_img_path + data + '_add.png', add)
                
                

                

    #         for label in self.label_list:
    #             if os.path.exists(self.label_path + post_img_path+ label + '.png'):

    #                 if not os.path.exists(self.output_label_path + match_par + '/' + post_img_path):
    #                     os.makedirs(self.output_label_path + match_par + '/' + post_img_path)

    #                 image = cv2.imread(self.label_path + post_img_path + label + '.png')
    #                 label_result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    #                 label_result[label_result == 0] = image [label_result == 0]
    #                 cv2.imwrite(self.output_label_path  + match_par+ '/' + post_img_path + label + '.png', label_result)     
    #     return shift


    def alignment(self,file_path):
        # patient_dict = read_patient_list(file_path)
        method_dict = {}
        image_folder = self.image_path+ '/1/'
        filenames = sorted(os.listdir(image_folder))
        for method in self.method_name:
            method_dict[method] = {}

        for filename in filenames: 
            if filename.endswith(".png"):

                patient_id, eye, date = filename.split('.png')[0].split('_')
                if patient_id not in method_dict[self.method_name[0]]:
                    post_treatment = ''
                    for method in self.method_name:
                        method_dict[method][patient_id] = {}
                if eye not in method_dict[self.method_name[0]][patient_id]:
                    pre_treatment =  date
                    for method in self.method_name:
                        method_dict[method][patient_id][eye] = {}
                else :
                    post_treatment =  date 
                for method in self.method_name:
                    method_dict[method][patient_id][eye][date] = {}
                treatment_patient = patient_id
                treatment_eye = eye

                

                if pre_treatment != '' and post_treatment != '':
                    shift = self.template_matching(treatment_patient, treatment_eye, pre_treatment, post_treatment)
                    print(shift)
                    method_dict[method][patient_id][eye][date] = shift
        return method_dict
                

    def template_matching(self, patient_id, eye, pre_treatment, post_treatment):
        pre_img = cv2.imread(self.image_path + '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
        post_img = cv2.imread(self.image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
        if pre_img is None or post_img is None:
            return
        gray_pre = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
        gray_post = cv2.cvtColor(post_img, cv2.COLOR_BGR2GRAY)
        shift = dict()

        for method in self.methods:
            match_par = self.method_name[self.methods.index(method)]
            
            
            # print method name
            shift[match_par]= []
            shift_x, shift_y = self.pointMatch(gray_pre, gray_post, method)
            shift[match_par] = [shift_x, shift_y]
            # 進行平移變換
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]]) # 平移矩陣


            for data in self.data_list:
                
                if os.path.exists(self.image_path + data + '/' + label + '_' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                    pre_image = cv2.imread(self.image_path + data + '/' + label + '_' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                    pre_image = cv2.resize(pre_image, self.image_size)
                    pre_image = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)
                    
                    cv2.imwrite(self.output_image_path+ match_par+ '/' + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_image)
                    if os.path.exists(self.image_path + data + '/' + label + '_' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                        image = cv2.imread(self.image_path + data + '/' + label + '_' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                        result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                        
                        if not os.path.exists(self.output_image_path  + match_par+ '/' + data ):
                            os.makedirs(self.output_image_path  + match_par+ '/' + data )
                        if not os.path.exists(self.output_image_path  + match_par+ '/' + data +'_move/'):
                            os.makedirs(self.output_image_path  + match_par+ '/' + data +'_move/')
                            

                        cv2.imwrite(self.output_image_path  + match_par+ '/' + data + '_move/'  + patient_id + '_' + eye + '_' + post_treatment + '.png', result)
                        result[result == 0] = pre_image [result == 0]
                        

                        cv2.imwrite(self.output_image_path  + match_par+ '/' + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)

                        # vis_img = result.copy()
                        # vis_img[:,:,0] = 0
                        # vis_img[:,:,2] = 0
                        # add = cv2.addWeighted(pre_image, 0.5, vis_img, 1.0, 0)
                        # if not os.path.exists(self.output_image_path  + match_par+ '/' + data + '_vis' ):
                        #     os.makedirs(self.output_image_path  + match_par+ '/' + data + '_vis' )
                        # cv2.imwrite(self.output_image_path  + match_par+ '/' + data + '_vis' + '/' + patient_id + '_' + eye + '_' + post_treatment +'_vis.png', add)


            for label in self.label_list:
                mask_path = os.path.join(os.path.dirname(os.path.dirname(self.image_path)),label,'masks')
                
                # label_path = os.path.join(path,'masks',label+ '_'+ )
                if os.path.exists(mask_path + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                    pre_label = cv2.imread(mask_path + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                    pre_label = cv2.resize(pre_label, self.image_size)
                    pre_label = cv2.normalize(pre_label, None, 0, 255, cv2.NORM_MINMAX)
                    # print('post_treatment',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                    cv2.imwrite(self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_label)                    
                    
                    if os.path.exists(mask_path + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                        post_label = cv2.imread(mask_path + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                        post_label = cv2.resize(post_label, self.image_size)
                        label_result = cv2.warpAffine(post_label, M, (image.shape[1], image.shape[0]))
                        
                        # print('post_treatment',self.output_label_path  + match_par+ '/' + label + '_move/'  + patient_id + '_' + eye + '_' + post_treatment + '.png',)
                        cv2.imwrite(self.output_label_path  + match_par+ '/' + label + '_move/'  + patient_id + '_' + eye + '_' + post_treatment + '.png', label_result)
                        label_result[label_result == 0] = pre_label [label_result == 0]
                        
                        if not os.path.exists(self.output_label_path  + match_par+ '/' + label ):
                            os.makedirs(self.output_label_path  + match_par+ '/' + label )
                        if not os.path.exists(self.output_label_path  + match_par+ '/' + label +'_move/'):
                            os.makedirs(self.output_label_path  + match_par+ '/' + label +'_move/')
                        # print('post_treatment',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                        cv2.imwrite(self.output_label_path  + match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', label_result)

        return shift
                     


def read_patient_list(path):
    f = open(path, 'r')
    patient = dict()
    patient_id = '' 
    for line in f.readlines():
        line = line.strip(' : \n')   
        if 'total' in line:
            break   
        if patient_id !='':
            if patient_id not in patient:
                patient[patient_id] = dict()
        line = line.split('\t')
        if(len(line) > 1):
            eyes = line[1].split(' : ')
            date = eyes[1].split(', ')
            if eyes[0] == 'OD':
                patient[patient_id]['R'] = date
            elif eyes[0] == 'OS':
                patient[patient_id]['L'] = date
        else :
            patient_id = line[0]
                # patient
    f.close()
    return patient

    

def setFolder(path):
    os.makedirs(path, exist_ok=True)




if __name__ == '__main__':
    # pre_treatment = "..\\..\\Data\\PPT\\00294362_20210511_1.png"
    # post_treatment = "..\\..\\Data\\PPT\\00294362_20221222_1.png"
    # matching().template_matching(pre_treatment, post_treatment)
    date = '1120'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    PATH_LABEL = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    image_path = PATH_BASE + 'ALL/'
    PATH_MATCH = image_path + 'MATCH/' 
    PATH_MATCH_LABEL = image_path + 'MATCH_LABEL/' 
    
    

    Match = template_matcher(image_path,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL)
    shift_patient_dict = Match.alignment('PCV_exist.txt')
    # json_file = './shift_patient_dict.json'
    # tools.write_to_json_file(json_file, shift_patient_dict)

    # evals= Match.avg_evaluate() 
    # json_file = './' + disease + '_' + date + '_eval.json'
    # tools.write_to_json_file(json_file, evals)

    # to csv
    # with open('./' + disease + '_' + date + '_eval.csv', 'w', newline='') as csvfile:
    #     # 建立 CSV 檔寫入器
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['method','mse','psnr','ssim'])
    #     for method, eval in evals.items():
    #         writer.writerow([method,eval['mse'],eval['psnr'],eval['ssim']])




    # original_eval= Match.get_pre_treatment_evaluation('PCV_exist.txt')
    # json_file = './original_eval.json'
    # tools.write_to_json_file(json_file, original_eval)

    #     path_patient = PATH_IMAGE + patient
    #     patient = patient
    #     Match.alignment(patient)
        # matching().alignment(path_patient, path_output, path_label, path_label_output)
        # break
        