import cv2 
import matplotlib.pyplot as plt
import numpy as np 


# 對齊可以看作是簡單的座標變換
# 將兩個影像都轉換為灰階影像
# 找到兩個影像的特徵點
# 將特徵點對齊



# Haris角點檢測
def harris_feature(img):
    gray = np.float32(img)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    return img

# Shi-Tomasi角點檢測
def shi_tomasi_feature(img):
    gray = np.float32(img)
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    return img







# Feature Matching FLANN
def flann_match(des1, des2 ):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) # 5
    search_params = dict(checks = 50) # 50
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    return matches

# Feature Matching BF
def bf_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    return matches

# Feature Matching Ratio Test
def ratio_test(matches):
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return good

# 獲取關鍵點的坐標
def get_coordinates(kp1, kp2, matches):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    return src_pts, dst_pts

# 計算轉換矩陣和變換後的圖像
def get_transform(src_pts, dst_pts):
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return M

# 計算變換後的圖像
def get_transform_img(img1, img2, M):
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    return img2


def registration(img1,img2,distance=0.8, method='SIFT'):
        # Initiate SIFT detector
    if method == 'SIFT':
        sift = cv2.SIFT_create()
    elif method == 'KAZE':
        sift = cv2.KAZE_create()
    elif method == 'AKAZE':
        sift = cv2.AKAZE_create()
    elif method == 'ORB':
        sift = cv2.ORB_create()
    elif method == 'BRISK':
        sift = cv2.BRISK_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) # 5
    search_params = dict(checks = 50) # 50

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)


    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < distance*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)

    # cv2.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3,),plt.show()

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    






if __name__ == '__main__':
    pre_treatment = "..\\..\\Data\\OCTA\\00200499\\20210125\\L\\1.png"
    post_treatment = "..\\..\\Data\\OCTA\\00200499\\20220124\\L\\1.png"
    img1 = cv2.imread(pre_treatment,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(post_treatment,cv2.IMREAD_GRAYSCALE)
    kp1, des1 = kaze_feature(img1)
    kp2, des2 = kaze_feature(img2)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    vis_img1 = img1.copy()
    vis_img2 = img2.copy()
    vis_img1 = cv2.drawKeypoints(vis_img1, kp1, None, color=(0, 255, 0), flags=0)
    vis_img1 = cv2.drawKeypoints(vis_img1, kp2, None, color=(0, 255, 0), flags=0)
    ax[0].imshow(vis_img1)
    ax[1].imshow(vis_img1)
    plt.show()

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(des1, des2, 2)
    good = ratio_test(matches)
    src_pts, dst_pts = get_coordinates(kp1, kp2, good)
    M = get_transform(src_pts, dst_pts)

    img3 = get_transform_img(img1, img2, M)

    add = cv2.addWeighted(img1, 0.5, img3, 0.5, 0)

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].imshow(img3)
    ax[1].imshow(add)
    plt.show()












    