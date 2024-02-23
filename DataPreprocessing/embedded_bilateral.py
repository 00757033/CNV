# import cv2
# import numpy as np

# def embedded_bilateral_filter(image, sigma_spatial, sigma_range):
#     height, width = image.shape
#     filtered_image = np.zeros_like(image, dtype=np.float32)

#     for y in range(height):
#         for x in range(width):
#             patch = image[max(0, y - 2 * sigma_spatial):min(height, y + 2 * sigma_spatial + 1),
#                           max(0, x - 2 * sigma_spatial):min(width, x + 2 * sigma_spatial + 1)]
            
#             # Calculate the Gaussian kernel weights for spatial and range domains
#             spatial_weights = np.exp(-((y - patch.shape[0] // 2)**2 + (x - patch.shape[1] // 2)**2) / (2 * sigma_spatial**2))
            
#             # Calculate the range weights with rescaled differences
#             range_differences = np.sum((patch - image[y, x])**2)
#             max_difference = np.max(range_differences)
#             scaled_differences = range_differences / max_difference  # Rescale the differences
#             range_weights = np.exp(-scaled_differences / (2 * (sigma_range**2)))
            
#             # Apply clamping to prevent overflow
#             spatial_weights = np.clip(spatial_weights, 0, 1)
#             range_weights = np.clip(range_weights, 0, 1)
            
#             # Compute the weighted average
#             normalization = np.sum(spatial_weights * range_weights)
            
#             # Handle the case when normalization is zero
#             if normalization != 0:
#                 filtered_image[y, x] = np.sum(patch * spatial_weights[..., np.newaxis] * range_weights[..., np.newaxis]) / normalization
#             else:
#                 filtered_image[y, x] = image[y, x]

#     return filtered_image.astype(np.uint8)

# # Read the input image
# image = cv2.imread('../../Data/images_info/INPAINT_TELEA03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)

# # Set the parameters (sigma_spatial and sigma_range) based on your application
# sigma_spatial = 50
# sigma_range = 10

# # Apply the embedded bilateral filter
# filtered_image = embedded_bilateral_filter(image, sigma_spatial, sigma_range)
# smooth = image - filtered_image
# # Display the original and filtered images
# cv2.imshow('Original Image', image)
# cv2.imshow('Embedded Bilateral Filtered Image', filtered_image)
# cv2.imshow('smooth', smooth)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2

# Load the input noisy infrared image
g = cv2.imread('../../Data/images_info/INPAINT_TELEA03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)

# Convert the input image to 32-bit float
g = g.astype(np.float32)

# Set parameters
lambda_value = 1024
sigma_spatial = 10  # Adjust as needed
sigma_range = 100  # Adjust as needed

# Calculate gradients of the input noisy image g
gradient_gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
gradient_gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

# Perform bilateral filtering separately on the x and y gradient components
bilateral_filtered_gx = cv2.bilateralFilter(gradient_gx, -1, sigma_spatial, sigma_range)
bilateral_filtered_gy = cv2.bilateralFilter(gradient_gy, -1, sigma_spatial, sigma_range)

# Calculate Zm for normalization
Zm_gx = cv2.bilateralFilter(np.ones_like(gradient_gx), -1, sigma_spatial, sigma_range)
Zm_gy = cv2.bilateralFilter(np.ones_like(gradient_gy), -1, sigma_spatial, sigma_range)

# Define the FFT and IFFT operators
def FFT(x):
    return fft2(x)

def IFFT(x):
    return ifft2(x)

# Solve the model using FFT and IFFT
u = IFFT(FFT(g) + lambda_value * (FFT(bilateral_filtered_gx) * FFT(bilateral_filtered_gx) + FFT(bilateral_filtered_gy) * FFT(bilateral_filtered_gy)) / (FFT(Zm_gx) + lambda_value * (FFT(gradient_gx) * FFT(gradient_gx) + FFT(gradient_gy) * FFT(gradient_gy))))

# Ensure the real part is used
u = np.real(u)

# Convert u to uint8 if needed
u = u.astype(np.uint8)

# Display or save the smoothed output image u
cv2.imshow('Noisy Image', g)
cv2.imshow('Smoothed Image', u)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the noisy infrared image 'g' (replace with your actual image path)


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the input noisy infrared image (g)
# g = cv2.imread('../../Data/need_label/PCV/4/03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)

# # Set parameters
# lambda_value = 1024.0  # Balance parameter

# # sigma_spatial in MATLAB value is 12.0 and sigma_range is 0.04
# # turn to python value
# sigma_spatial = 12
# sigma_range = 10


# # Apply bilateral filtering to gradients of the noisy image
# def bilateral_filter_gradients(input_image, sigma_spatial, sigma_range):
#     gradient_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)
#     # Convert gradients to 32f
#     gradient_x = gradient_x.astype(np.float32)
#     gradient_y = gradient_y.astype(np.float32)
#     bilateral_x = cv2.bilateralFilter(gradient_x, d=-1, sigmaColor=sigma_range, sigmaSpace=sigma_spatial)
#     bilateral_y = cv2.bilateralFilter(gradient_y, d=-1, sigmaColor=sigma_range, sigmaSpace=sigma_spatial)

#     return bilateral_x, bilateral_y

# # Calculate bilateral filtered gradients
# bilateral_gradient_x, bilateral_gradient_y = bilateral_filter_gradients(g, sigma_spatial, sigma_range)

# # Compute the normalization factor (Zm)
# def calculate_normalization_factor(input_image, m, neighborhood, sigma_spatial, gradient_mq):
#     normalization_factor = 0.0
#     for n in neighborhood:
#         diff = m - n
#         spatial_weight = np.exp(-np.sum(diff ** 2) / (2 * sigma_spatial ** 2))
#         range_diff = gradient_mq - input_image[n]
#         range_weight = np.exp(-(range_diff ** 2) / (2 * sigma_range ** 2))
#         normalization_factor += spatial_weight * range_weight
#     return normalization_factor

# # Initialize the output image as a 3D array
# output_image = np.zeros((g.shape[0], g.shape[1], 2), dtype=np.float32)  # 2 channels for x and y gradients
# height, width = g.shape[:2]

# # Iterate through each pixel to calculate the output_image
# for y in range(height):
#     for x in range(width):
#         m = np.array([y, x])
#         neighborhood = [(i, j) for i in range(max(0, y - 1), min(height, y + 2)) for j in range(max(0, x - 1), min(width, x + 2))]

#         # Calculate normalization factors for x and y gradients
#         normalization_x = calculate_normalization_factor(bilateral_gradient_x, m, neighborhood, sigma_spatial, gradient_mq=bilateral_gradient_x[y, x])
#         normalization_y = calculate_normalization_factor(bilateral_gradient_y, m, neighborhood, sigma_spatial, gradient_mq=bilateral_gradient_y[y, x])

#         # Calculate the smoothed pixel value for each channel (x and y gradients)
#         for channel in range(2):
#             weighted_sum = 0.0
#             for n in neighborhood:
#                 diff = m - n
#                 spatial_weight = np.exp(-np.sum(diff ** 2) / (2 * sigma_spatial ** 2))
#                 range_diff = bilateral_gradient_x[y, x] - bilateral_gradient_x[n[0], n[1]] if channel == 0 else bilateral_gradient_y[y, x] - bilateral_gradient_y[n[0], n[1]]
#                 range_weight = np.exp(-(range_diff ** 2) / (2 * sigma_range ** 2))
#                 weighted_sum += spatial_weight * range_weight * g[n[0], n[1]]

#             normalization = normalization_x if channel == 0 else normalization_y
#             output_image[y, x, channel] = weighted_sum / normalization

# # Apply FFT and IFFT to solve the model (Eq. 3)
# def solve_blf_ls_model(input_image, bilateral_gradient_x, bilateral_gradient_y, lambda_value):
#     input_fft = np.fft.fft2(input_image)
#     gradient_x_fft = np.fft.fft2(bilateral_gradient_x)
#     gradient_y_fft = np.fft.fft2(bilateral_gradient_y)
    
#     # Calculate the output image using the updated gradients
#     output_image = np.zeros_like(input_image, dtype=np.float32)
#     height, width = input_image.shape[:2]

#     for y in range(height):
#         for x in range(width):
#             # Calculate the smoothed pixel value for each channel (x and y gradients)
#             weighted_sum = 0.0
#             normalization_x = 0.0
#             normalization_y = 0.0
#             m = np.array([y, x])
#             neighborhood = [(i, j) for i in range(max(0, y - 1), min(height, y + 2)) for j in range(max(0, x - 1), min(width, x + 2))]

#             for channel in range(2):
#                 for n in neighborhood:
#                     diff = m - np.array(n)
#                     spatial_weight = np.exp(-np.sum(diff ** 2) / (2 * sigma_spatial ** 2))
#                     range_diff = bilateral_gradient_x[y, x] - bilateral_gradient_x[n[0], n[1]] if channel == 0 else bilateral_gradient_y[y, x] - bilateral_gradient_y[n[0], n[1]]
#                     range_weight = np.exp(-(range_diff ** 2) / (2 * sigma_range ** 2))
#                     weighted_sum += spatial_weight * range_weight * input_image[n[0], n[1]]
#                     if channel == 0:
#                         normalization_x += spatial_weight * range_weight
#                     else:
#                         normalization_y += spatial_weight * range_weight

#                 normalization = normalization_x if channel == 0 else normalization_y
#                 output_image[y, x] = weighted_sum / normalization

#     # Continue with the calculation of output_fft
#     denominator = np.fft.fft2(np.ones_like(input_image)) + lambda_value * (np.fft.fft2(gradient_x_fft.conjugate() * gradient_x_fft) + np.fft.fft2(gradient_y_fft.conjugate() * gradient_y_fft))
#     output_fft = (input_fft + lambda_value * (gradient_x_fft.conjugate() * np.fft.fft2(output_image) + gradient_y_fft.conjugate() * np.fft.fft2(output_image))) / denominator
#     output_image = np.fft.ifft2(output_fft).real
#     return output_image

# # Solve the BLF-LS model to obtain the smoothed output image (u)
# smoothed_output_image = solve_blf_ls_model(g, bilateral_gradient_x, bilateral_gradient_y, lambda_value)

# # fft to time domain
# smoothed_output_image = np.fft.ifft2(smoothed_output_image).real


# # display original image and its histogram
# plt.figure(figsize=(10, 10))
# plt.subplot(3, 2, 1)
# plt.imshow(g, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(3, 2, 2)
# plt.hist(g.ravel(), 256, [0, 256])
# plt.title('Original Image Histogram')
# plt.xlabel('Intensity')
# plt.ylabel('Count')

# # display smoothed output image and its histogram
# plt.subplot(3, 2, 3)
# plt.imshow(smoothed_output_image, cmap='gray')
# plt.title('Smoothed Output Image')
# plt.axis('off')

# plt.subplot(3, 2, 4)
# plt.hist(smoothed_output_image.ravel(), 256, [0, 256])
# plt.title('Smoothed Output Image Histogram')
# plt.xlabel('Intensity')
# plt.ylabel('Count')

# # high frequency image (g - u) and its histogram
# plt.subplot(3, 2, 5)
# plt.imshow(g - smoothed_output_image, cmap='gray')
# plt.title('High Frequency Image')
# plt.axis('off')

# plt.subplot(3, 2, 6)
# plt.hist((g - smoothed_output_image).ravel(), 256, [0, 256])
# plt.title('High Frequency Image Histogram')
# plt.xlabel('Intensity')
# plt.ylabel('Count')

# plt.show()


# import cv2
# import numpy as np

# # 加载图像
# image = cv2.imread('../../Data/need_label/PCV/4/03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)

# # 双边滤波参数
# d = 2  # 空间域参数
# sigma_space = 10  # 空间范围参数
# sigma_color = 10   # 色彩范围参数

# # 使用双边滤波平滑梯度
# smoothed_gradient = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# # 最小二乘法参数
# lambda_value = 0.01  # 正则化参数

# # BLF-LS滤波
# def blf_ls_filter(image, smoothed_gradient, lambda_value):
#     smooth_image = np.zeros_like(image, dtype=np.float32)
#     height, width = image.shape[:2]

#     for y in range(height):
#         for x in range(width):
#             gradient_value = smoothed_gradient[y, x]
#             pixel_value = image[y, x]

#             # 计算局部窗口
#             y_min = max(0, y - d)
#             y_max = min(height - 1, y + d)
#             x_min = max(0, x - d)
#             x_max = min(width - 1, x + d)

#             # 初始化变量
#             weighted_sum = 0
#             weight_normalization = 0

#             for j in range(y_min, y_max + 1):
#                 for i in range(x_min, x_max + 1):
#                     neighbor_gradient_value = smoothed_gradient[j, i]
#                     neighbor_pixel_value = image[j, i]

#                     # 计算空间和范围权重
#                     spatial_weight = np.exp(-((y - j) ** 2 + (x - i) ** 2) / (2 * sigma_space ** 2))
#                     # Calculate the range difference without squaring and convert to float
#                     range_difference = np.float32(neighbor_gradient_value) - np.float32(gradient_value)
#                     range_weight = np.exp(-(range_difference ** 2) / (2 * sigma_color ** 2))



#                     # 计算加权和和权重归一化
#                     weighted_sum += spatial_weight * range_weight * neighbor_pixel_value
#                     weight_normalization += spatial_weight * range_weight

#             # 更新平滑图像
#             smooth_image[y, x] = weighted_sum / weight_normalization

#     return smooth_image.astype(np.uint8)



# # 应用BLF-LS滤波
# filtered_image = blf_ls_filter(image, smoothed_gradient, lambda_value)

# # high frequency image
# smooth = image - filtered_image


# # 显示原始图像
# cv2.imshow('Original Image', image)
# # 显示平滑梯度
# cv2.imshow('Smoothed Gradient', smoothed_gradient)
# # 显示BLF-LS滤波结果
# cv2.imshow('Filtered Image', filtered_image)
# cv2.imshow('smooth', smooth)


# 

# import cv2
# import numpy as np
# import scipy.fftpack as fft
# import matplotlib.pyplot as plt

# # Define the parameters
# lambda_value = 1024
# sigma_s = 2
# sigma_r = 0.04

# # Define the FFT and IFFT operators
# def fft2(x):
#     return fft.fftshift(fft.fft2(fft.ifftshift(x)))

# def ifft2(x):
#     return fft.fftshift(fft.ifft2(fft.ifftshift(x)))

# # Define the geometric closure function G(sigma_s)
# def geometric_closure(m, n, sigma_s):
#     distance = np.linalg.norm(m - n)
#     return np.exp(-distance**2 / (2 * sigma_s**2))

# # Define the intensity similarity function G(sigma_r)
# def intensity_similarity(grad_m, grad_n, sigma_r):
#     diff = grad_m - grad_n
#     return np.exp(-np.linalg.norm(diff)**2 / (2 * sigma_r**2))

# # Define the bilateral filter fBLF(∇g[q])
# def bilateral_filter(grad_g, sigma_s, sigma_r):
#     rows, cols,channels= grad_g.shape
#     filtered_grad = np.zeros_like(grad_g)

#     for i in range(rows):
#         for j in range(cols):
#             for c in range(channels):
#                 N_t = []  # Neighborhood centered on the pixel (i, j)

#                 for m in range(i - 1, i + 2):
#                     for n in range(j - 1, j + 2):
#                         if 0 <= m < rows and 0 <= n < cols:
#                             N_t.append((m, n))

#                 Z_m = 0
#                 result = np.zeros(grad_g[i, j].shape, dtype=grad_g.dtype)

#                 for n in N_t:
#                     weight_s = geometric_closure(np.array([i, j]), np.array(n), sigma_s)
#                     weight_r = intensity_similarity(grad_g[i, j, :], grad_g[n[0], n[1], :], sigma_r)
#                     weight = weight_s * weight_r

#                     Z_m += weight
#                     result += weight * grad_g[n[0], n[1]]

#                 filtered_grad[i, j] = result / Z_m

#     return filtered_grad

# # Define the main BLF-LS function
# def blf_ls(input_image):
#     # Compute the image gradient ∇g
#     grad_g_x = np.gradient(input_image, axis=(1, 0))
#     grad_g_y = np.gradient(input_image, axis=(0, 1))
#     grad_g = np.stack([grad_g_x, grad_g_y], axis=-1)

#     # Calculate the magnitude of the gradients
#     magnitude_grad_g = np.linalg.norm(grad_g, axis=-1)

#     # Apply bilateral filtering to the magnitude of the gradients
#     filtered_magnitude_grad_g = bilateral_filter(magnitude_grad_g, sigma_s, sigma_r)

#     # Calculate the FFT of the input and filtered gradients
#     input_fft = fft2(magnitude_grad_g)
#     filtered_grad_fft = fft2(filtered_magnitude_grad_g)

#     # Solve the BLF-LS equation using FFT and IFFT
#     epsilon = 1e-10  # Small epsilon value
#     denominator = input_fft + lambda_value * filtered_grad_fft
#     u = ifft2(denominator / (fft2(np.ones_like(magnitude_grad_g)) + lambda_value * fft2(np.ones_like(filtered_magnitude_grad_g)) + epsilon))

#     return np.abs(u)

# # Load your noisy infrared image using OpenCV
# image_path = '../../Data/need_label/PCV/4/03959415_R_20220714.png'
# g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Apply the BLF-LS algorithm to the loaded image
# smoothed_image = blf_ls(g)

# # Compute the high-frequency image (g - u)
# high_freq_image = g - smoothed_image

# # Display the original image and its histogram
# plt.figure(figsize=(12, 6))
# plt.subplot(131)
# plt.imshow(g, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(132)
# plt.hist(g.ravel(), bins=256, range=(0, 256), density=True, color='blue', alpha=0.6)
# plt.title('Histogram of Original Image')
# plt.xlabel('Pixel Value')
# plt.ylabel('Normalized Frequency')

# # Display the smoothed output image and its histogram
# plt.subplot(133)
# plt.imshow(smoothed_image, cmap='gray')
# plt.title('Smoothed Image')
# plt.axis('off')

# plt.figure(figsize=(12, 6))
# plt.subplot(131)
# plt.hist(smoothed_image.ravel(), bins=256, range=(0, 256), density=True, color='green', alpha=0.6)
# plt.title('Histogram of Smoothed Image')
# plt.xlabel('Pixel Value')
# plt.ylabel('Normalized Frequency')

# # Display the high-frequency image and its histogram
# plt.figure(figsize=(12, 6))
# plt.subplot(131)
# plt.imshow(high_freq_image, cmap='gray')
# plt.title('High-Frequency Image (g - u)')
# plt.axis('off')

# plt.subplot(132)
# plt.hist(high_freq_image.ravel(), bins=256, range=(-128, 128), density=True, color='red', alpha=0.6)
# plt.title('Histogram of High-Frequency Image')
# plt.xlabel('Pixel Value')
# plt.ylabel('Normalized Frequency')

# plt.tight_layout()
# plt.show()











