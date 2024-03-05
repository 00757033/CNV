from msilib.schema import Binary
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology
import statistics
from scipy import stats
import scipy.misc
import pandas as pd 


def vessel_feature_area(img_g, msk_g):
    #lab_g = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(msk_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    msk_contours = cv2.drawContours(msk_g.copy()*255, contours, -1, (0, 0, 255), 3)
    size = 50
    #msk_rm_small = morphology.remove_small_objects(msk_g*255, size)
    if len(contours) != 0:
        # 集合所有contours的點，並找出影像的序號
        points = np.vstack(contours)
        # 計算面積
        area = cv2.contourArea(points)
        # 計算圓率
        ellipse = cv2.fitEllipse(points)
        # 紀錄中心點，用於計算位移
        center=ellipse[0]
        center_1 = round(center[0])
        center_2 = round(center[1])
        center = (center_1, center_2)
        #cv2.circle(img_g, center, 1, (255,255,255), 5)
        x, y, w, h = cv2.boundingRect(points)
        frame=img_g[y:y+h, x:x+w]
        area_ratio = area/(h*w)
        img_mask = img_g
        img_mask[msk_g==0] = 0
        
        frame_mask=img_mask #[y:y+h, x:x+w]  
        if area > 0:
            brigthness = sum(sum(frame_mask))/area
        else:
            brigthness = 0
    else:
        area_ratio = 0
        center = 0
        brigthness = 0
        frame_mask = 0
        h = 0
        w = 0
        area = 0
    
    return area_ratio, center, brigthness, frame_mask, h*w

def vessel_feature_analysis(image, mask):
    binary = otsu(image)
    binary[binary==255] = 1
    binary[mask==0] = 0
    total_size = np.size(image)
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8)*255
    size = 1000
    binary = binary*255
    image_count = np.sum(mask)
    binary_count = np.sum(np.array(binary) >= 1)
    skel_count = np.sum(np.array(skeleton) >= 1)

    if total_size == 0:
        VD = 0
        VLD = 0
        VDI = 0
    else:
        VD = binary_count/total_size
        VLD = skel_count/total_size
    if skel_count == 0:
        VDI = 0
    else :
        VDI = binary_count/skel_count
    
    print('image_count', total_size)
    print('binary_count',binary_count)
    print('skel_count', skel_count)
    return binary, skeleton, VD, VLD, VDI

def otsu(image, kernal_size = (3,3)):
        image = cv2.GaussianBlur(image,kernal_size,0)
        try:
            ret3,th3 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        except:
            print('NO')
            th3 = np.zeros((304,304))
        return th3

def feature_extract(address, image_floder, label_floder):
    areas = []
    centers = []
    brigths = []
    frames_mask = []
    frames_bi = []
    frame_skel = []
    VD_list = []
    VLD_list = []
    VDI_list = []
    images = os.listdir(address + '/' + image_floder)
    masks = os.listdir(address + '/' + label_floder)
    setFloder(address + 'skel')
    setFloder(address + 'binary')
    for i in range(len(images)):
        image = cv2.imread(address + '/' + image_floder + '/' + images[i])   
        mask = cv2.imread(address + '/' + label_floder + '/' + masks[i])
        img_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        msk_g = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        area_ratio, center, brigthness, frame_mask, square= vessel_feature_area(img_g, msk_g)
        image_bi, image_skel, VD, VLD, VDI = vessel_feature_analysis(frame_mask, msk_g)
  
        print('Area : ', area_ratio)
        print('center : ', center)
        print('brigthness : ', brigthness)
        print('square', square)
        print('---------------------------------')
        cv2.imwrite(address + '/' + 'skel' + '/' + images[i], image_skel)
        cv2.imwrite(address + '/' + 'binary' + '/' + images[i], image_bi)
        areas.append(area_ratio)
        centers.append(center)
        brigths.append(brigthness)
        frames_mask.append(frame_mask)
        frames_bi.append(image_bi)
        frame_skel.append(image_skel)
        VD_list.append(VD)
        VLD_list.append(VLD)
        VDI_list.append(VDI)
    return areas, VD_list, VLD_list, VDI_list, images

def setFloder(path):
    if not os.path.isdir(path) : os.mkdir(path)

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
    plt.subplot(1,3, number+1)
    plt.bar(1.4, score_pre, width, color='dimgrey', label='Pre-injection', yerr=[[lower_error], [pre_std]] , capsize=5, linewidth=1, edgecolor=['black'], zorder=100)
    plt.bar(1.4+width*1.3, score_post, width, color='lightgrey', label='Post-injection', yerr=[[lower_error], [post_std]], capsize=5, linewidth=1, edgecolor=['black'],zorder=100)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    #plt.xlabel('Degree')
    plt.ylabel(func, fontsize=15)
    plt.yticks(fontsize=10)
    plt.ylim(0, Max*1.6)
    #plt.title(func)
    plt.legend(loc=1, frameon=False, fontsize=10).set_zorder(200)

def drawBar(scores, std, number, func):
    
    width = 0.3
    lower_error = [0]
    Max = 0
    Min = 0
    if Max < np.max(scores) + np.max(std):
        Max = np.max(scores) + np.max(std)
    if Min > np.min(scores):
        Min = np.min(scores) - np.min(std)
    plt.subplot(1,3, number+1)
    if(Min >= 0):
        plt.bar(1.4, [scores], width, color='dimgrey', yerr=[lower_error, [std]] , capsize=5, linewidth=1, edgecolor=['black'])
        plt.ylim(Min, Max*1.6)
    else:
        plt.bar(1.4, [scores], width, color='dimgrey', yerr=[[std], lower_error] , capsize=5, linewidth=1, edgecolor=['black'])
        plt.ylim(Min*1.6, Max)

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    #plt.xlabel('Degree')
    plt.ylabel(func, fontsize=14)
    #plt.title(func)
    plt.legend(loc=1, frameon=False).set_zorder(200)

def getMeanStd(feature):
    data_mean = []
    data_std = []
    for i in feature:
        data_mean.append(round(sum(i)/len(i), 4))
        data_std.append(round(statistics.pstdev(i), 4))
    
    return data_mean, data_std
    
def getDiff(feature1, feature2):
    data = []
    data_diff = []
    data_mean = []
    data_std  = []
    for i in range(0, len(feature1)):
        for j in range(0, len(feature1[0])):
            data.append((feature2[i][j]-feature1[i][j])/feature1[i][j])
        data_diff.append(data)
        data = []
    for i in data_diff:
        data_mean.append(round(sum(i)/len(i), 4))
        data_std.append(round(statistics.pstdev(i), 4))

    return data_diff, data_mean, data_std

def main():
    PATH_BASE = '../../Data/'
    data_class = 'CNV'
    data_date = '0831'
    data_count = 70
    PATH_BASE  =  PATH_BASE + data_class + '_' + data_date + '/'
    data_type = 'OR'
    path_pre  = PATH_BASE + data_type + '/compare/1/'
    path_post = PATH_BASE + data_type + '/compare/2/'
    image_floder = 'image'
    label_floder = 'label'
    areas_pre, VD_list_pre, VLD_list_pre, VDI_list_pre, names_pre = feature_extract(path_pre, image_floder, label_floder)
    areas_post,  VD_list_post, VLD_list_post, VDI_list_post, names_post  = feature_extract(path_post, image_floder, label_floder)
    data_length_pre = len(areas_pre)
    data_count = data_count*2
    feature_pre = [VD_list_pre, VLD_list_pre, VDI_list_pre]
    feature_post = [ VD_list_post, VLD_list_post, VDI_list_post]


    feature_name = ['Vessel Density', 'Vessel Legth Density', 'Vessel Diameter Index']
    pre_mean, pre_std = getMeanStd(feature_pre)
    post_mean, post_std = getMeanStd(feature_post)
    data_diff, data_diff_mean, data_diff_std = getDiff(feature_pre, feature_post)

    dict = {'VD_pre': VD_list_pre, 'VD_post': VD_list_post, 'VLD_pre': VLD_list_pre, 'VLD_post': VLD_list_post, 'VDI_pre': VDI_list_pre, 'VDI_post': VDI_list_post} 
    df = pd.DataFrame(dict) 
    df.to_csv(data_date + '_VesselFeature_' + data_type + '.csv')


    f = open('vesselAnalysis_' + data_class + '_' + data_date + '_' + data_type + '.txt', 'a')
    for i in range(0,len(names_pre)):
        lines = [names_pre[i]+ '\n']
        f.writelines(lines)
        lines = ['Vessel Density ', 'pre : ' + str(round(VD_list_pre[i], 5)), '  post : ' + str(round(VD_list_post[i], 5))+ '\n']
        f.writelines(lines)
        lines = ['Vessel Legth Density ', 'pre : ' + str(round(VLD_list_pre[i], 5)), '  post : ' + str(round(VLD_list_post[i], 5))+ '\n']
        f.writelines(lines)
        lines = ['----------------------------------------------------------\n']
        f.writelines(lines)
    f.close()  

    #plt.figure(figsize=(10, 10))
    plt.figure()
    for i in range(0,3):
        t, p = stats.ttest_rel(feature_pre[i], feature_post[i])
        t2, p2 = stats.ttest_ind_from_stats(pre_mean[i], pre_std[i], data_count, post_mean[i],post_std[i], data_count)
        print(feature_name[i], " pre : ", pre_mean[i], ', std : ', pre_std[i])
        print(feature_name[i], " post : ", post_mean[i], ', std : ', post_std[i])
        print(feature_name[i], " change ratio : ", 100*round(( post_mean[i]-pre_mean[i])/pre_mean[i], 3))
        print(feature_name[i], " Diff: ", data_diff_mean[i], ', std : ', data_diff_std[i])
        print('資料數量:' + str(data_length_pre))
        print("P-value : " + str(round(p2, 5)))
        #print("P-value : " + str(round(p, 5)))
        drawBarCompare(pre_mean[i], post_mean[i], pre_std[i], post_std[i], i, feature_name[i])
    plt.show()
    
if __name__ == '__main__':
    main()
