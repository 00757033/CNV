import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle, vortex
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
import cv2
import os
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
def optical_flow(image0,image1):
    # image0 = plt.imread(image0)
    # image1 = plt.imread(image1)

    # --- Load the sequence
    # image0, image1, disp = stereo_motorcycle()

    # --- Convert the images to gray level: color is not supported.
    # image0 = rgb2gray(image0)
    # image1 = rgb2gray(image1)

    # --- Compute the optical flow
    # v, u = optical_flow_ilk(image0, image1, radius  = 5)
    v, u = optical_flow_tvl1(image0, image1, attachment=0.8)
    # --- Use the estimated optical flow for registration

    nr, nc = image0.shape

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

    image1_warp = warp(image1, np.array([row_coords + v, col_coords + u]), mode='edge')

    # # --- Display the result
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 8))
    
    ax[0].imshow(image0, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Reference image')
    
    ax[1].imshow(image1, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Target image')
    
    ax[2].imshow(image1_warp, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('Warped image')

    
    

    plt.show()
    
    return image1_warp



if __name__ == '__main__':
    date = '20240502'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    label_path = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    output_image_path = PATH_BASE + 'ALL/'
    image_path = output_image_path + 'MATCH/' 
    output_label_path = output_image_path + 'MATCH_LABEL/' 
    image_folder =output_image_path+ '/1/'
    filenames = sorted(os.listdir(image_folder))
    
    patient_dict = {}
    MSE = []
    PSNR = []
    SSIM = []
    original_MSE = []
    original_PSNR = []
    original_SSIM = []
    
    for filename in filenames: 
        if filename.endswith(".png"):
            print(filename)

            patient_id, eye, date = filename.split('.png')[0].split('_')

            if patient_id + '_' + eye not in patient_dict :
                patient_dict[patient_id + '_' + eye] = {}
                pre_treatment = date
                post_treatment = ''
            else :
                post_treatment =  date 

                treatment_patient = patient_id
                treatment_eye = eye
    
                pre_treatment_image = image_folder + patient_id + '_' + eye + '_' + str(pre_treatment) + '.png'
                post_treatment_image = image_folder + patient_id + '_' + eye + '_' + str(post_treatment) + '.png'
                
                image1 = cv2.imread(pre_treatment_image,cv2.IMREAD_GRAYSCALE)
                image2 = cv2.imread(post_treatment_image, cv2.IMREAD_GRAYSCALE)
                
                image1_warp = optical_flow(image1,image2)
                
                # evaluation
                # MSE
                mse = np.mean((image1 - image1_warp) ** 2)
                # PSNR
                psnr = peak_signal_noise_ratio(image1, image1_warp)
                
                # SSIM
                ssim = structural_similarity(image1, image1_warp)
                
                MSE.append(mse)
                PSNR.append(psnr)
                SSIM.append(ssim)
                
                # original
                original_mse = np.mean((image1 - image2) ** 2)
                original_psnr = peak_signal_noise_ratio(image1, image2)
                original_ssim = structural_similarity(image1, image2)
                
                original_MSE.append(original_mse)
                original_PSNR.append(original_psnr)
                original_SSIM.append(original_ssim)
                

    print('MSE:', np.mean(MSE))
    print('PSNR:', np.mean(PSNR))
    print('SSIM:', np.mean(SSIM))
    print('original_MSE:', np.mean(original_MSE))
    print('original_PSNR:', np.mean(original_PSNR))
    print('original_SSIM:', np.mean(original_SSIM))
    
                
                
                
                
                
    
