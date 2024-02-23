import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import statistics
from scipy import stats
import scipy.misc

import os
import glob

PATH_BASE = '../../Data/'
label_dir = 'Label'
PATH_LABEL = PATH_BASE + label_dir + '/'
PATH_IMAGE = PATH_BASE + 'OCTA/' 

#Template Match Method:找到template在img中最高的相關係數的數值(r)和位置(top_left)
def getPoints(img, template, method=cv2.TM_CCOEFF_NORMED):
    result = cv2.matchTemplate(img, template, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        r = min_val
    else:
        top_left = max_loc
        r = max_val
    return top_left, r #top_left:回傳對位左上角的位置, r=最高的相關係數

#將術後的影像切成4個角落跟中央，並分別跟術前影像做getPoints，找到5個template中R最高的位置，並回傳要位移的x跟y
def pointMatch(image, template):
    template_total      = template
    template_center     = template[52:52+199, 52:52+199]
    template_left_up    = template[0:199, 0:199]
    template_right_up   = template[104:104+199, 0:199]
    template_left_down  = template[0:199, 104:104+199]
    template_right_down = template[104:104+199, 104:104+199]

    elements_total,  r_t  = getPoints(image, template_total     , method=cv2.TM_CCOEFF_NORMED)
    elements_center, r_c  = getPoints(image, template_center    , method=cv2.TM_CCOEFF_NORMED)
    elements_c = (elements_center[0]-52, elements_center[1]-52)

    elements_1,      r_1  = getPoints(image, template_left_up   , method=cv2.TM_CCOEFF_NORMED)
    elements_TL = (elements_1[0], elements_1[1])
    elements_2,      r_2  = getPoints(image, template_right_up  , method=cv2.TM_CCOEFF_NORMED)
    elements_TR = (elements_2[0], elements_2[1]-104)
    elements_3,      r_3  = getPoints(image, template_left_down , method=cv2.TM_CCOEFF_NORMED)
    elements_DL = (elements_3[0]-104, elements_3[1])
    elements_4,      r_4  = getPoints(image, template_right_down, method=cv2.TM_CCOEFF_NORMED)
    elements_DR = (elements_4[0]-104, elements_4[1]-104)

    print('Total : ', round(r_t,3) , " shift(x,y) :" + str(elements_total))
    print('Center : ', round(r_c,3), " shift(x,y) :" + str(elements_c))
    print('TL : ', round(r_1,3), " shift(x,y) :" + str(elements_TL))
    print('TR : ', round(r_2,3), " shift(x,y) :" + str(elements_TR))
    print('DL : ', round(r_3,3), " shift(x,y) :" + str(elements_DL))
    print('DR : ', round(r_4,3), " shift(x,y) :" + str(elements_DR))
    e = [elements_total, elements_c, elements_TL, elements_TR, elements_DL, elements_DR]
    r = [r_t, r_c, r_1, r_2, r_3, r_4]
    Relation = 0
    shift_x = None
    shift_y = None
    for i in range(0, len(r)):
        if r[i] > Relation:
            Relation = r[i]
            shift_x = e[i][0]
            shift_y = e[i][1]
    if shift_x != None : 
        return shift_x, shift_y
    else:
        print('No Match')

def setFloder(path):
    if not os.path.isdir(path) : os.mkdir(path)
#建立對位後輸出的資料夾
def createFolder(path_output, date_list, eye):
    if not os.path.isdir(path_output) : os.mkdir(path_output)
    if not os.path.isdir(path_output + '/' + date_list[0]) : os.mkdir(path_output + '/' + date_list[0])
    if not os.path.isdir(path_output + '/' + date_list[1]) : os.mkdir(path_output + '/' + date_list[1])
    if not os.path.isdir(path_output + '/' + date_list[0] + '/' + eye ) : os.mkdir(path_output + '/' + date_list[0] + '/' + eye)
    if not os.path.isdir(path_output + '/' + date_list[1] + '/' + eye) : os.mkdir(path_output + '/' + date_list[1] + '/' + eye)

#對位並比較相似度參數，將對位後的資料跟label寫出
def alignment(path_patient, path_output, path_label, path_label_output):
    print('Patient : ', path_patient[-8:])
    SSIM_origin = 0
    MSE_origin = 0
    PSNR_origin = 0

    SSIM_match = 0
    MSE_match = 0
    PSNR_match = 0

    date_list = os.listdir(path_patient)
    data_list = ['2', '3', '4', '1_OCT', '2_OCT', '3_OCT', '4_OCT'] #要對位的影像
    label_list = ['OR_CNV', 'CC_CNV']
    eye_list = ['R', 'L']
    if len(date_list) > 1:
        path_pre_date = path_patient + '/' + date_list[0] + '/'
        path_post_date = path_patient + '/' + date_list[1] + '/'
        for eye in eye_list:
            if os.path.exists(path_pre_date + eye) and os.path.exists(path_post_date + eye):
                image_pre = cv2.imread(path_pre_date + eye + '/1.png', 0)
                image_post = cv2.imread(path_post_date + eye + '/1.png', 0)
                image_pre = cv2.resize(image_pre, (304,304))
                image_post = cv2.resize(image_post, (304,304))
                print(image_pre.shape, image_post.shape)
                shift_x, shift_y = pointMatch(image_pre, image_post) #找到治療前後的位移
                translation_matrix = np.float32([ [1,0,shift_x], [0,1,shift_y] ])
                img_post_match_1 = cv2.warpAffine(image_post, translation_matrix, (304,304)) 
                img_pre_match_1  = image_pre
                if shift_x >=0:
                    img_pre_match_1[:, :shift_x] = 0
                else:
                    img_pre_match_1[:, shift_x:] = 0
                if shift_y >=0:
                    img_pre_match_1[:shift_y, :] = 0
                else:
                    img_pre_match_1[shift_y:, :] = 0
                print(path_output + '/' + date_list[0] + '/' + eye + '/1.png')
                cv2.imwrite(path_output + '/' + date_list[0] + '/' + eye + '/1.png', img_pre_match_1)
                cv2.imwrite(path_output + '/' + date_list[1] + '/' + eye + '/1.png', img_post_match_1)

                SSIM_origin = structural_similarity(image_pre, image_post, multichannel=True)
                MSE_origin = mean_squared_error(image_pre, image_post)
                PSNR_origin = peak_signal_noise_ratio(image_pre, image_post)

                SSIM_match = structural_similarity(img_pre_match_1, img_post_match_1, multichannel=True)
                MSE_match = mean_squared_error(img_pre_match_1, img_post_match_1)
                PSNR_match = peak_signal_noise_ratio(img_pre_match_1, img_post_match_1)

                createFolder(path_output, date_list, eye)
                #讀取其他層的OCT跟OCTA影像對位後寫出
                for data in data_list:
                    if os.path.exists(path_pre_date + eye + '/' + data + '.png') and os.path.exists(path_post_date + eye + '/' + data + '.png'):
                        image_pre_other = cv2.imread(path_pre_date + eye + '/' + data + '.png', 0)
                        image_post_other = cv2.imread(path_post_date + eye + '/' + data + '.png', 0)
                        image_pre_other = cv2.resize(image_pre_other, (304,304))
                        image_post_other = cv2.resize(image_post_other, (304,304))
                        translation_matrix = np.float32([ [1,0,shift_x], [0,1,shift_y] ])
                        img_post_match = cv2.warpAffine(image_post_other, translation_matrix, (304,304)) 
                        img_pre_match  = image_pre_other
                        if shift_x >=0:
                            img_pre_match[:, :shift_x] = 0
                        else:
                            img_pre_match[:, shift_x:] = 0
                        if shift_y >=0:
                            img_pre_match[:shift_y, :] = 0
                        else:
                            img_pre_match[shift_y:, :] = 0
                        cv2.imwrite(path_output + '/' + date_list[0] + '/' + eye + '/' + data + '.png', img_pre_match)
                        cv2.imwrite(path_output + '/' + date_list[1] + '/' + eye + '/' + data + '.png', img_post_match)
                #讀取label對位後寫出
                for label in label_list:
                    if os.path.exists(path_label + '/' + date_list[0] + '/' + eye + '/' + label + '.png'):
                        createFolder(path_label_output, date_list, eye)
                        image_pre_other = cv2.imread(path_label + '/' + date_list[0] + '/' + eye + '/' + label + '.png', 0)
                        image_pre_other = cv2.resize(image_pre_other, (304,304))
                        if os.path.exists(path_label + '/' + date_list[1] + '/' + eye + '/' + label + '.png'):
                            image_post_other = cv2.imread(path_label + '/' + date_list[1] + '/' + eye + '/' + label + '.png', 0)
                            image_post_other = cv2.resize(image_post_other, (304,304))
                        else:
                            image_post_other = np.zeros((304,304))
                        translation_matrix = np.float32([ [1,0,shift_x], [0,1,shift_y] ])
                        img_post_match = cv2.warpAffine(image_post_other, translation_matrix, (304,304)) 
                        img_pre_match  = image_pre_other
                        if shift_x >= 0:
                            img_pre_match[:, :shift_x] = 0
                        else:
                            img_pre_match[:, shift_x:] = 0
                        if shift_y >= 0:
                            img_pre_match[:shift_y, :] = 0
                        else:
                            img_pre_match[shift_y:, :] = 0
                        cv2.imwrite(path_label_output + '/' + date_list[0] + '/' + eye + '/' + label + '.png', img_pre_match*255)
                        cv2.imwrite(path_label_output + '/' + date_list[1] + '/' + eye + '/' + label + '.png', img_post_match*255)

                print('原始影像SSIM : ' , SSIM_origin)
                print('MATCH後SSIM : ' , SSIM_match)

                print('原始影像MSE : ' , MSE_origin)
                print('MATCH後MSE : ' , MSE_match)

                print('原始影像PSNR : ' , PSNR_origin)
                print('MATCH後PSNR : ' , PSNR_match)

    return SSIM_origin, SSIM_match, MSE_origin, MSE_match, PSNR_origin, PSNR_match
#計算影像相似度標準差跟平均
def getMeanStd(feature):
    data_mean = []
    data_std = []
    for i in feature:
        data_mean.append(round(sum(i)/len(i), 4))
        data_std.append(round(statistics.pstdev(i), 4))
    
    return data_mean, data_std
#將對位結果畫成barchart
def drawBarCompare(pre_scores, post_scores, pre_std, post_std, number, func):
    score_pre = [pre_scores]
    score_post = [post_scores]
    x = np.array([1.4, 2])
    width = 0.01
    lower_error = 0
    Max = 0
   
    if Max < np.max(post_scores) + np.max(post_std):
        Max = np.max(post_scores) + np.max(post_std)
    if Max < np.max(pre_scores) + np.max(pre_std):
        Max = np.max(pre_scores) + np.max(pre_std)
    plt.subplot(1,5, number+1)
    plt.bar(1.4, score_pre, width, color='dimgrey', label='Original dataset', yerr=[[lower_error], [pre_std]] , capsize=5, linewidth=1, edgecolor=['black'], zorder=100)
    plt.bar(1.4+width*1.3, score_post, width, color='lightgrey', label='Aligned dataset', yerr=[[lower_error], [post_std]], capsize=5, linewidth=1, edgecolor=['black'],zorder=100)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    #plt.xlabel('Degree')
    plt.ylabel(func, fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0, Max*1.6)
    #plt.title(func)
    plt.legend(loc=1, frameon=False, fontsize=10).set_zorder(200)


def main():
    PATH_BASE = '../../Data/'
    label_dir = 'Label'
    PATH_LABEL = PATH_BASE + label_dir + '/'
    PATH_IMAGE = PATH_BASE + 'OCTA/' 
    PATH_MATCH = PATH_BASE + 'MATCH' 
    PATH_MATCH_LABEL = PATH_BASE + 'MATCH_LABEL' 
    setFloder(PATH_MATCH)
    setFloder(PATH_MATCH_LABEL)
    patient_list = os.listdir(PATH_IMAGE)
    SSIM_origin_list = []
    SSIM_match_list  = []
    MSE_origin_list  = []
    MSE_match_list   = []
    PSNR_origin_list = []
    PSNR_match_list  = []
    for patient in patient_list:       
        SSIM_origin, SSIM_match, MSE_origin, MSE_match, PSNR_origin, PSNR_match = alignment(PATH_IMAGE + patient, PATH_MATCH + '/' + patient, PATH_LABEL + patient, PATH_MATCH_LABEL + '/' + patient)
        if SSIM_origin != 0 or MSE_origin !=0 or PSNR_origin !=0 : 
            SSIM_origin_list.append(SSIM_origin)
            SSIM_match_list.append(SSIM_match)
            MSE_origin_list.append(MSE_origin)
            MSE_match_list.append(MSE_match)
            PSNR_origin_list.append(PSNR_origin)
            PSNR_match_list.append(PSNR_match)
    feature_name= ['SSIM', 'MSE', 'PSNR']
    feature_pre = [SSIM_origin_list, MSE_origin_list, PSNR_origin_list]
    feature_post = [SSIM_match_list, MSE_match_list, PSNR_match_list]
    feature_pre_mean, feature_pre_std = getMeanStd(feature_pre)
    feature_post_mean, feature_post_std = getMeanStd(feature_post)
    data_count = len(SSIM_origin_list)
    for i in range(0,3):
        t, p = stats.ttest_rel(feature_pre[i], feature_post[i])
        t2, p2 = stats.ttest_ind_from_stats(feature_pre_mean[i], feature_pre_std[i], data_count, feature_post_mean[i],feature_post_std[i], data_count)
        print(feature_name[i], " pre : ", feature_pre_mean[i], ', std : ', feature_pre_std[i])
        print(feature_name[i], " post : ", feature_post_mean[i], ', std : ', feature_post_std[i])
        print(feature_name[i], " change ratio : ", 100*round(( feature_post_mean[i]-feature_pre_mean[i])/feature_pre_mean[i], 3))
        print("P-value : " + str(round(p2, 5)))
        drawBarCompare(feature_pre_mean[i], feature_post_mean[i], feature_pre_std[i], feature_post_std[i], i, feature_name[i])
    plt.show()
    plt.close()


if __name__ == '__main__': 
    main()   
           

