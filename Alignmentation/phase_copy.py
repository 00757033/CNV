import cv2
import numpy as np
import os
import match as mt
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import time
import shutil
import tools.tools as tools
import pathlib as pl
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
def setFolder(path):
    os.makedirs(path, exist_ok=True)

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


def phase_correlation(gray1, gray2):

        gray1=  cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
        gray2=  cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)
        # Convert to floating-point images
        gray1 = np.float32(gray1)
        gray2 = np.float32(gray2)

        # 計算相位相關
        result = cv2.phaseCorrelate(gray1, gray2)

        # 變換為位移量
        shift = [result[0][0], result[0][1]]

        return shift

class phase_correlate():
    def __init__(self,path_image,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL,PATH_MATCH_IMAGE):
        self.path_image = path_image
        self.path_label = PATH_LABEL
        self.path_match = PATH_MATCH
        self.data_list = ['1','2', '3', '4', '1_OCT', '2_OCT', '3_OCT', '4_OCT']
        self.label_list = ['label_3', 'label_4']
        self.methods = 'PHASE_CORRELATE'
        self.path_match_image = PATH_MATCH_IMAGE
        self.output_path = PATH_MATCH + self.methods + '/'
        self.output_label_path = PATH_MATCH_LABEL + self.methods + '/'
        setFolder(PATH_MATCH)
        
        setFolder(PATH_MATCH + '/' + self.methods)
        setFolder(PATH_MATCH_LABEL)
        setFolder(PATH_MATCH_LABEL + '/' + self.methods)

        for data in self.data_list:
            setFolder(self.output_path +  data)
            setFolder(self.output_path+ data)





    def alignment(self, patient_id, eye, pre_treatment, post_treatment):
        print(self.path_match_image + '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
        print(self.path_match_image + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
        img1 = cv2.imread(self.path_match_image + '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
        img1 = cv2.resize(img1, (304, 304))
        img2 = cv2.imread(self.path_match_image + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
        img2 = cv2.resize(img2, (304, 304))
        shift = self.phase_correlation(img1, img2)
    
        print(shift)
        # 對位
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        for data in self.data_list:
            
            if os.path.exists(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                pre_image = cv2.imread(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                cv2.imwrite(self.output_path  + '/' + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_image)
            if os.path.exists(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                image = cv2.imread(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                if not os.path.exists(self.output_path + data + '_shift/'):
                    os.makedirs(self.output_path + data + '_shift/')
                cv2.imwrite(self.output_path + data + '_shift/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)
                result[result == 0] = image [result == 0]

                cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)
        
                vis_img = result.copy()
                vis_img[:,:,0] = 0
                vis_img[:,:,2] = 0
                image_pre = cv2.imread(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                image_pre = cv2.resize(image_pre, (304, 304))
                vis_img = cv2.addWeighted(image_pre, 0.5, vis_img, 0.5, 0)
                if not os.path.exists(self.output_path + data + '_vis/'):
                    os.makedirs(self.output_path + data + '_vis/')



                cv2.imwrite(self.output_path + data + '_vis/' + patient_id + '_' + eye + '_' + post_treatment + '_vis.png', vis_img)

        
        return shift


    def phase_correlation(self,image1, image2):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Apply Fourier Transform
        f1 = np.fft.fft2(gray1)
        f2 = np.fft.fft2(gray2)

        # Calculate cross power spectrum
        cross_power_spectrum = np.multiply(f1, np.conj(f2))

        # Calculate phase correlation
        phase_correlation = np.fft.ifft2(cross_power_spectrum)
        phase_correlation = np.fft.fftshift(phase_correlation)

        # Calculate magnitude and angle
        magnitude = np.abs(phase_correlation)
        angle = np.angle(phase_correlation)

        # Find the peak in the magnitude
        _, _, _, max_loc = cv2.minMaxLoc(magnitude)

        # Calculate translation (shift)
        rows, cols = gray2.shape
        center = (cols // 2, rows // 2)
        shift = [max_loc[0] - center[0], max_loc[1] - center[1]]

        return shift

    def phase(self):
        patient_dict = {}
        image_folder = self.path_match_image+ '/1/'
        filenames = sorted(os.listdir(image_folder))

        for filename in filenames: 
            if filename.endswith(".png"):

                patient_id, eye, date = filename.split('.png')[0].split('_')
                if patient_id not in patient_dict:
                    patient_dict[patient_id] = {}
                    post_treatment = ''
                if eye not in patient_dict[patient_id]:
                    patient_dict[patient_id][eye] = {}

                    pre_treatment =  date

                else :
                    patient_dict[patient_id][eye][date] = []
                    post_treatment =  date 

                    treatment_patient = patient_id
                    treatment_eye = eye

                    print(treatment_patient, treatment_eye, pre_treatment, post_treatment)

                    if pre_treatment != '' and post_treatment != '':
                        shift = self.alignment(treatment_patient, treatment_eye, pre_treatment, post_treatment)
                        patient_dict[patient_id][eye][post_treatment]=shift

        return patient_dict

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

        return mse,psnr,ssim

    def all_evaluate(self,match_path):
        eval = {}
        original = {}

        for method in pl.Path(match_path).iterdir():
            print(method)
            if method.is_dir():
                method = method.name

                eval[method] = {}
                patient = {}
                image_folder = match_path+ '/1/'
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
                matching_mse_list = []
                matching_psnr_list = []
                matching_ssim_list = []
                for filename in filenames: 
                    if filename.endswith(".png"):
                        patient_id, eye, date = filename.split('.png')[0].split('_')
                        if patient_id + '_' + eye not in patient :
                            patient[patient_id + '_' + eye] = {}
                            post_treatment = ''
                            pre_treatment = date
                            patient[patient_id + '_' + eye]['pre_treatment'] = date
                        else :
                            patient[patient_id + '_' + eye]['post_treatment'] = {}
                            patient[patient_id + '_' + eye]['post_treatment']['date'] = date
                            post_treatment =  date 

                            if patient[patient_id + '_' + eye]['pre_treatment'] != '' and patient[patient_id + '_' + eye]['post_treatment'] != '':

                                pre_treatment_img = self.path_match_image+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                                post_treatment_img = self.path_match_image + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                                matching_img = self.path_match + method + '/1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                                mse,psnr,ssim = self.evaluate(pre_treatment_img,post_treatment_img)
                                mse_list.append(mse)
                                psnr_list.append(psnr)
                                ssim_list.append(ssim)

                                patient[patient_id + '_' + eye]['post_treatment']['original'] = [mse,psnr,ssim]

                                matching_mse,matching_psnr,matching_ssim = self.evaluate(pre_treatment_img,matching_img)
                                matching_mse_list.append(matching_mse)
                                matching_psnr_list.append(matching_psnr)
                                matching_ssim_list.append(matching_ssim)

                                patient[patient_id + '_' + eye]['post_treatment']['matching'] = [matching_mse,matching_psnr,matching_ssim]

                                if matching_ssim - ssim > best_differece_ssim :
                                    best_differece_ssim = matching_ssim - ssim
                                    best_case['patient'] = [ patient_id,eye,post_treatment]
                                    best_case ['psnr'] = psnr
                                    best_case ['ssim'] = ssim
                                    best_case ['mse'] = mse
                                    best_case ['matching_psnr'] = matching_psnr
                                    best_case ['matching_ssim'] = matching_ssim
                                    best_case ['matching_mse'] = matching_mse

                                if matching_ssim - ssim < worst_differece_ssim :
                                    worst_differece_ssim = matching_ssim - ssim
                                    worst_case['patient'] = [ patient_id,eye,post_treatment]
                                    worst_case ['psnr'] = psnr
                                    worst_case ['ssim'] = ssim
                                    worst_case ['mse'] = mse
                                    worst_case ['matching_psnr'] = matching_psnr
                                    worst_case ['matching_ssim'] = matching_ssim
                                    worst_case ['matching_mse'] = matching_mse


                matching_avg_mse = round(sum(matching_mse_list)/len(matching_mse_list),3)
                matching_avg_psnr = round(sum(matching_psnr_list)/len(matching_psnr_list),3)
                matching_avg_ssim = round(sum(matching_ssim_list)/len(matching_ssim_list),3)
                matching_mse_std = round(np.std(matching_mse_list, ddof=1),3)
                matching_psnr_std = round(np.std(matching_psnr_list, ddof=1),3)
                matching_ssim_std = round(np.std(matching_ssim_list, ddof=1),3)

                patient['avg'] = {}
                patient['avg']['mse'] = [matching_avg_mse,matching_mse_std]
                patient['avg']['psnr'] = [matching_avg_psnr,matching_psnr_std]
                patient['avg']['ssim'] = [matching_avg_ssim,matching_ssim_std]

                eval[method] = patient['avg']


                if best_case != {} :
                    patient['best_case'] = best_case
                    eval[method]['best_case'] = best_case
                if worst_case != {} :
                    patient['worst_case'] = worst_case
                    eval[method]['worst_case'] = worst_case


                
                    avg_mse = round(sum(mse_list)/len(mse_list),3)
                    avg_psnr = round(sum(psnr_list)/len(psnr_list),3)
                    avg_ssim = round(sum(ssim_list)/len(ssim_list),3)
                    mse_std = round(np.std(mse_list, ddof=1),3)
                    psnr_std = round(np.std(psnr_list, ddof=1),3)
                    ssim_std = round(np.std(ssim_list, ddof=1),3)

                    patient['original'] = {}
                    patient['original']['mse'] = [avg_mse,mse_std]
                    patient['original']['psnr'] = [avg_psnr,psnr_std]
                    patient['original']['ssim'] = [avg_ssim,ssim_std]


                if original == {} :
                    eval['original'] = patient['original']
                
                # save patient to csv





        return eval







if __name__ == '__main__':
    date = '1121'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    PATH_LABEL = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    PATH_MATCH_IMAGE = PATH_DATA + 'PCV_1121/ALL/'
    PATH_MATCH = PATH_MATCH_IMAGE + 'MATCH/' 
    PATH_MATCH_LABEL = PATH_MATCH_IMAGE + 'MATCH_LABEL/' 

    phase = phase_correlate(PATH_IMAGE,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL,PATH_MATCH_IMAGE)
    phase_dict = phase.phase()
    json_file = './phase_patient_dict.json'
    tools.write_to_json_file(json_file, phase_dict)




    




