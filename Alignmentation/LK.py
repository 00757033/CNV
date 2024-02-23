# Lucas-Kanade

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time



# distance=0.8, method='SIFT',matcher='BF'
# distance=0.8, method='KAZE',matcher='BF'
# distance=0.8, method='ORB',matcher='BF'
def LK(img1, img2, kp1, kp2, des1, des2, distance=0.8, method='SIFT',matcher='FLANN'):
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
    elif method == 'BRIEF':
        detector = cv2.FastFeatureDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    

    elif method == 'FREAK':
        detector = cv2.FastFeatureDetector_create()
        freak = cv2.xfeatures2d.FREAK_create()

    
    if method == 'FREAK':
        kp1 = detector.detect(img1)
        kp2 = detector.detect(img2)
        kp1, des1 = freak.compute(img1, kp1)
        kp2, des2 = freak.compute(img2, kp2)
    elif method == 'BRIEF':
        kp1 = detector.detect(img1)
        kp2 = detector.detect(img2)
        kp1, des1 = brief.compute(img1, kp1)
        kp2, des2 = brief.compute(img2, kp2)



    else:

        kp1 = sift.detect(img1, None)
        kp2 = sift.detect(img2, None)


        kp1, des1 = sift.compute(img1, kp1)
        kp2, des2 = sift.compute(img2, kp2)

    # # Find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    pre = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    post = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(pre)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(post)
    plt.axis('off')
    plt.show()

    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    # Matching method
    if method == 'SIFT'  or method == 'BRISK' or method == 'BRIEF':
        if  matcher == 'BF':
            bf = cv2.BFMatcher( cv2.NORM_L2)
            matches = bf.knnMatch(des1, des2, k=2)

        elif matcher == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
    elif method == 'ORB':
        if  matcher == 'BF':
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

        elif matcher == 'FLANN':
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1) #2
            search_params = dict(checks=100)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
    elif method == 'KAZE' or method == 'AKAZE' or method == 'SURF' or method == 'FREAK':
        if  matcher == 'BF':
            bf = cv2.BFMatcher( cv2.NORM_L2)
            matches = bf.knnMatch(des1, des2, k=2)

        elif matcher == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)


    matches = sorted(matches, key=lambda x: x[0].distance)
    min_dist = matches[0][0].distance




    # Apply ratio test
    good = []
    pts1 = []
    pts2 = []
    MIN_COUNT = 3
    for i, (m, n) in enumerate(matches): # m: best match, n: second best match
        if m.distance < distance* n.distance :
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        
        if m.distance > 2*min_dist:
            break
    if len(good) < MIN_COUNT:
        print("Error: Not enough matches.")
        print("Translation: ", None)
        print("Rotation angle: ", None)
        print("Scale: ", None)

        return None


    img3 = cv2.drawMatchesKnn(img1 , kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3)
    plt.axis('off')
    plt.show()

    # Assuming pts1 and pts2 are lists of corresponding points
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    # Check for non-empty point sets
    if len(pts1) == 0 or len(pts2) == 0:
        print("Error: Empty point sets.")
        print("Translation: ", None)
        print("Rotation angle: ", None)
        print("Scale: ", None)

        return img1 , None

    # print(pts1)
    # print(pts2)
    # print(len(pts1))
    # print(len(pts2))
    # 變換矩陣 
    # Find the transformation matrix using RANSAC
    H, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.LMEDS, ransacReprojThreshold=5.0)

    if H is None:
        print("Error: Homography is None.")
        print("Translation: ", None)
        print("Rotation angle: ", None)
        print("Scale: ", None)
        return img1 , None

    translation = (H[0, 2], H[1, 2])
    rotation_angle = np.arctan2(H[1, 0], H[0, 0])  # angle in radians
    scale = H[0, 0] / np.cos(rotation_angle)  # scale factor

    print("Translation: ", translation)
    print("Rotation angle: ", rotation_angle)
    print("Scale: ", scale)

    new_H = np.array( [[  scale*np.cos(rotation_angle), scale*np.sin(rotation_angle), translation[0]],
                        [ -scale*np.sin(rotation_angle), scale*np.cos(rotation_angle), translation[1]],
                        [ 0, 0, 1]], dtype=np.float32)






    # Use homography
    height, width= img2.shape
    im1Reg = cv2.warpPerspective(img1, new_H, (width, height))




    return im1Reg, H

if __name__ == '__main__':
    pre_treatment = "..\\..\\Data\\PCV_1017\\ALL\\1\\08532554_R_20170830.png"
    post_treatment = "..\\..\\Data\\PCV_1017\\ALL\\1\\08532554_R_20171206.png"
    img1 = cv2.imread(pre_treatment,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(post_treatment, cv2.IMREAD_GRAYSCALE)


    start = time.time()
    im1Reg, H = LK(img2, img1, None, None, None, None)
    end = time.time()
    print("LK time: ", end - start)
    
    add = cv2.addWeighted(img1, 0.5, im1Reg, 0.5, 0)
    vis_img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    vis_img2 = cv2.cvtColor(im1Reg, cv2.COLOR_GRAY2BGR)
    vis_img2[:, :, 0] = 0
    vis_img2[:, :, 2] = 0
    add = cv2.addWeighted(vis_img, 0.5, vis_img2, 0.5, 0)
    im1Reg[im1Reg == 0] = img1[im1Reg == 0]
    # Display images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Pre-treatment')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Post-treatment')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(im1Reg, cmap='gray')
    plt.title('LK')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(add, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')




    plt.show()

    

