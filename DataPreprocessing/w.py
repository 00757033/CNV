# import numpy as np
# import cv2
# from matplotlib import pyplot as plt



# # 读取OCTA图像
# octa_image = cv2.imread("..\..\Data\PCV_0915\CC\images\CC_08532554_R_20180103.png", cv2.IMREAD_GRAYSCALE)

# # # 执行小波变换
# # coeffs = pywt.wavedec2(octa_image, 'db1', level=1)  # 选择小波基和分解级别

# # # 对小波系数进行阈值处理以去除噪声
# # threshold = 200  # 根据需要调整阈值
# # coeffs = [pywt.threshold(c, threshold, mode='soft') if isinstance(c, np.ndarray) else c for c in coeffs]

# # # 重构图像
# # denoised_image = pywt.waverec2(coeffs, 'db1')

# # # subtract the smoothed image from the input image
# # sub = octa_image - denoised_image

# # # 显示结果
# # cv2.imshow('Original Image', octa_image)
# # cv2.imshow('Denoised Image', denoised_image)
# # cv2.imshow('sub', sub)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # f_transform = np.fft.fft2(octa_image)
# # f_shift = np.fft.fftshift(f_transform)
# # magnitude_spectrum = np.log(np.abs(f_shift) + 1)
# # rows, cols = octa_image.shape
# # crow, ccol = rows // 2, cols // 2  # 中心点坐标

# # # 创建高通滤波器
# # mask = np.ones((rows, cols), np.uint8)
# # mask[crow-30:crow+30, ccol-30:ccol+30] = 0  # 选择保留的频率区域
# # f_filtered = f_shift * mask
# # f_filtered_shift = np.fft.ifftshift(f_filtered)
# # image_filtered = np.fft.ifft2(f_filtered_shift)
# # image_filtered = np.abs(image_filtered)

# # sub = np.abs(octa_image - image_filtered)
# # plt.subplot(131), plt.imshow(octa_image, cmap='gray')
# # plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# # plt.subplot(132), plt.imshow(image_filtered, cmap='gray')
# # plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

# # plt.subplot(133), plt.imshow(sub, cmap='gray')
# # plt.title('sub'), plt.xticks([]), plt.yticks([])

# # plt.show()
# # 应用非局部均值滤波
# import numpy as np
# import cv2

# # 載入 OCTA 影像
# octa_image = cv2.imread("..\..\Data\PCV_0915\CC\images\CC_08532554_R_20180103.png", cv2.IMREAD_GRAYSCALE)
# # 執行傅立葉變換
# f_transform = np.fft.fft2(octa_image)
# f_transform_shifted = np.fft.fftshift(f_transform)

# # 設定高通濾波器的閾值
# threshold = 5
# rows, cols = octa_image.shape
# center_row, center_col = rows // 2, cols // 2

# #  保留中心區域 去除高頻部分
# f_transform_shifted[center_row - threshold:center_row + threshold, center_col - threshold:center_col + threshold] = 0



# # 執行高通濾波
# f_transform_unshifted = np.fft.ifftshift(f_transform_shifted)
# filtered_image = np.fft.ifft2(f_transform_unshifted)
# filtered_image = np.abs(filtered_image)



# # 調整影像的對比度和亮度（可選）
# filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
# filtered_image = filtered_image.astype(np.uint8)

# # subtract the smoothed image from the input image
# sub = octa_image - filtered_image 
# sub = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)

# # 顯示和儲存結果
# cv2.imshow('Original OCTA Image', octa_image)
# cv2.imshow('Filtered OCTA Image', filtered_image)
# cv2.imshow('sub', sub)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 加载OCTA图像（假设是一个多帧的三维数组）
octa_data = cv2.imread("..\..\Data\PCV_0915\CC\images\CC_08532554_R_20180103.png", cv2.IMREAD_GRAYSCALE)

import numpy as np
import cv2

def remove_stripes_octa(octa_image):
    # Convert OCTA image to float32
    octa_image = octa_image.astype(np.float32)

    # Compute the 1D FFT along the columns (assuming stripes are vertical)
    fft_result = np.fft.fft(octa_image, axis=0)

    # Create a mask to suppress low-frequency components (stripes)
    mask = np.ones_like(fft_result)
    rows, cols = octa_image.shape
    strip_width = 30  # Adjust this value based on the width of the stripes
    mask[:strip_width, :] = 0
    mask[rows-strip_width:, :] = 0

    # Apply the mask to the FFT result
    fft_result_filtered = fft_result * mask

    # Compute the inverse FFT to get the destriped image
    destriped_image = np.fft.ifft(fft_result_filtered, axis=0).real

    # Normalize the output image to the range [0, 255] and convert to uint8
    destriped_image = (destriped_image - np.min(destriped_image)) / (np.max(destriped_image) - np.min(destriped_image)) * 255
    destriped_image = destriped_image.astype(np.uint8)

    # image
    sub = np.abs(octa_image - destriped_image)

    return destriped_image , sub

import numpy as np
import cv2

def destripe_octa_image(octa_image, sigma_s=1, sigma_r=1):
    # Convert OCTA image to floating point for processing
    octa_image = octa_image.astype(np.float32)

    # Apply a bilateral filter to the OCTA image to smooth out noise while preserving edges
    # smoothed_image = cv2.bilateralFilter(octa_image, -1, sigma_s, sigma_r)

    # display
    cv2.imshow('smoothed_image', octa_image)
    cv2.waitKey(0)

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



# Load the OCTA image
octa_image = cv2.imread("..\..\Data\PCV_0915\OR\images\OR_01836108_L_20170320.png", cv2.IMREAD_GRAYSCALE)

# Apply destriping
destriped_octa  = destripe_octa_image(octa_image)

# PSNR
psnr = cv2.PSNR(octa_image, destriped_octa)
print(psnr)

# Detect potential missing or noisy areas (adjust the threshold as needed)
mask = destriped_octa < 5  # Example threshold, adjust accordingly

# Use cv2.inpaint to fill the detected areas
# inpainted_image = cv2.inpaint(destriped_octa, mask.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Display the original OCTA image, destriped image, and inpainted image
cv2.imshow('Original OCTA Image', octa_image)
cv2.imshow('Destriped OCTA Image', destriped_octa)

cv2.waitKey(0)
cv2.destroyAllWindows()