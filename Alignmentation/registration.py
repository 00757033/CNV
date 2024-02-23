import cv2 
import matplotlib.pyplot as plt
import numpy as np 
# 對齊可以看作是簡單的座標變換
# 將兩個影像都轉換為灰階影像
# 找到兩個影像的特徵點
# 將特徵點對齊
# https://www.geeksforgeeks.org/image-registration-using-opencv-python/
# Open the image files. 

ref_img = cv2.imread("..\\..\\Data\\OCTA\\00294362\\20210511\L\\1.png") # Reference image.
img = cv2.imread("..\\..\\Data\\OCTA\\00294362\\20221222\\L\\1.png") # Image to be aligned.
# cv2.imshow("Reference Image", ref_img)
# cv2.imshow("Image to be aligned", img)
# cv2.waitKey(0)

# Convert images to grayscale
ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 找到特徵點
# Harris Corner Detection : Harris 角點偵測是一種經典的角點偵測方法，它使用影像中的像素灰階變化來偵測角點。
harris_corners = cv2.cornerHarris(ref_img_gray, 3, 3, 0.05) # 3, 3 是窗口大小，0.05 是角點偵測的參數
harris_corners = cv2.dilate(harris_corners, None) # 用來擴張影像中的白色區域
# Threshold for an optimal value, it may vary depending on the image.
harris = ref_img.copy()
harris[harris_corners > 0.025 * harris_corners.max()] = [0,0,255]
# cv2.imshow("Harris Corners", harris)
# cv2.waitKey(0)

# shi-tomasi 角點偵測 : 它使用影像中的局部區域的特徵值（如最小特徵值）來確定角點位置。
shi_tomasi_corners = cv2.goodFeaturesToTrack(ref_img_gray, 200, 0.01, 30)
# 用來畫出角點
shi = ref_img.copy()
for corner in shi_tomasi_corners:
    x, y = corner[0]
    x = int(x)
    y = int(y)
    cv2.circle(shi, (x, y), 5, (0,0,255), -1)
# cv2.imshow("Shi-Tomasi Corners", shi)
# cv2.waitKey(0)

# FAST 特徵檢測器 : 它是一種檢測器，用於檢測影像中的角點。
fast = cv2.FastFeatureDetector_create()
# 找到關鍵點
keypoints = fast.detect(ref_img_gray, None)
# 用來畫出關鍵點
kp = ref_img.copy()
cv2.drawKeypoints(ref_img, keypoints, kp, color=(255,0,0))
# cv2.imshow("Key Points", kp)
# cv2.waitKey(0)

# ORB 特徵檢測器 : 它是一種檢測器，用於檢測影像中的角點。
orb = cv2.ORB_create()
# 找到關鍵點
keypoints_orb = orb.detect(ref_img_gray, None)
# 用來畫出關鍵點
kp_orb = ref_img.copy()
cv2.drawKeypoints(ref_img, keypoints_orb, kp_orb, color=(255,0,0))
# cv2.imshow("Key Points", kp_orb)
# cv2.waitKey(0)

# SIFT 特徵檢測器 : 它是一種檢測器，用於檢測影像中的角點。
sift = cv2.xfeatures2d.SIFT_create()
# 找到關鍵點
keypoints_sift = sift.detect(ref_img_gray, None)
# 用來畫出關鍵點
kp_sift = ref_img.copy()
cv2.drawKeypoints(ref_img, keypoints_sift, kp_sift, color=(255,0,0))
# cv2.imshow("Key Points", kp_sift)
# cv2.waitKey(0)

# # SURF 特徵檢測器 : 它是一種檢測器，用於檢測影像中的角點。
# surf = cv2.xfeatures2d.SURF_create(800)
# # 找到關鍵點
# keypoints_surf = surf.detect(ref_img_gray, None)
# # 用來畫出關鍵點
# kp_surf = ref_img.copy()
# cv2.drawKeypoints(ref_img, keypoints_surf, kp_surf, color=(255,0,0))
# cv2.imshow("Key Points", kp_surf)
# cv2.waitKey(0)

# BRISK 特徵檢測器 : 它是一種檢測器，用於檢測影像中的角點。
brisk = cv2.BRISK_create()
# 找到關鍵點
keypoints_brisk = brisk.detect(ref_img_gray, None)
# 用來畫出關鍵點
kp_brisk = ref_img.copy()
cv2.drawKeypoints(ref_img, keypoints_brisk, kp_brisk, color=(255,0,0))
# cv2.imshow("Key Points", kp_brisk)
# cv2.waitKey(0)

# plt 顯示圖片
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs[0, 0].imshow(ref_img)
axs[0, 0].set_title('Reference Image')
axs[0, 1].imshow(harris)
axs[0, 1].set_title('Harris Corners')
axs[0, 2].imshow(shi)
axs[0, 2].set_title('Shi-Tomasi Corners')
axs[1, 0].imshow(kp)
axs[1, 0].set_title('FAST Features')
axs[1, 1].imshow(kp_orb)
axs[1, 1].set_title('ORB Features')
axs[1, 2].imshow(kp_sift)
axs[1, 2].set_title('SIFT Features')
axs[0, 3].imshow(kp_brisk)
axs[0, 3].set_title('BRISK Features')
axs[1, 3].imshow(ref_img)
axs[1, 3].set_title('SURF Features')

plt.show()







