# import cv2
# import numpy as np

# def blf_ls(input_image, lambda_val=1024,sigmaColor=75,sigmaSpace=75):
#     # Bilateral filter the input image
#     filtered_image = cv2.bilateralFilter(input_image, d=0, sigmaColor = sigmaColor, sigmaSpace = sigmaSpace)
    
#     # Compute image gradients
#     gradient_x = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 0, ksize=5)
#     gradient_y = cv2.Sobel(filtered_image, cv2.CV_64F, 0, 1, ksize=5)
    
#     # Solve the BLF-LS equation using NumPy
#     A = lambda_val * (gradient_x ** 2 + gradient_y ** 2) + 1
#     B = lambda_val * (gradient_x * cv2.filter2D(input_image, -1, gradient_x) + gradient_y * cv2.filter2D(input_image, -1, gradient_y))
#     u = B / A
    
#     return u

# # Load the grayscale image 'g'
# g = cv2.imread('../../Data/need_label/PCV/4/03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)

# # Apply the BLF-LS algorithm to smooth the image
# smoothed_image = blf_ls(g, lambda_val=1024, sigmaColor=100, sigmaSpace=20)



# # Display the original image 
# cv2.imshow('Original Image', g)
# # Display the smoothed image 
# cv2.imshow('Smoothed Image', g -smoothed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2
# import numpy as np

# def fBLF(gradient_g, m, n, sigma_s=2, sigma_r=0.04):
#     Nt = np.exp(-(np.linalg.norm(m - n) ** 2) / (2 * sigma_s ** 2))
#     Ir = np.exp(-(np.linalg.norm(gradient_g[m] - gradient_g[n]) ** 2) / (2 * sigma_r ** 2))
#     return Nt * Ir * gradient_g[n]

# def blf_ls(input_image, lambda_value=1024, sigma_s=2, sigma_r=4):
#     gradient_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=5)
#     gradient_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=5)
#     gradient_g = np.dstack((gradient_x, gradient_y))
#     u = input_image.copy()
    
#     for i in range(5):  # You can adjust the number of iterations
#         for x in range(input_image.shape[0]):
#             for y in range(input_image.shape[1]):
#                 m = np.array([x, y])
#                 neighborhood = input_image[max(0, x-1):min(input_image.shape[0], x+2), 
#                                            max(0, y-1):min(input_image.shape[1], y+2)]
#                 fblf_x = np.sum([fBLF(gradient_g, m, n, sigma_s, sigma_r)[0] for n in neighborhood.flatten()])
#                 fblf_y = np.sum([fBLF(gradient_g, m, n, sigma_s, sigma_r)[1] for n in neighborhood.flatten()])
#                 u[x, y] = (lambda_value * (gradient_x[x, y] * fblf_x + gradient_y[x, y] * fblf_y) + input_image[x, y]) / (lambda_value * (gradient_x[x, y] ** 2 + gradient_y[x, y] ** 2) + 1)
    
#     return u

# # Load your noisy infrared image here
# input_image = cv2.imread('../../Data/need_label/PCV/4/03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)

# # Apply BLF-LS smoothing
# output_image = blf_ls(input_image)

# # Display the result or save it to a file
# cv2.imshow('Smoothed Image', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2
# import numpy as np

# def blf_ls(input_image, lambda_value=1024):
#     # Define constants
#     sigma_s = 2
#     sigma_r = 0.04

#     # Bilateral filter the input image
#     smoothed_image = cv2.bilateralFilter(input_image, d=0, sigmaColor=sigma_r, sigmaSpace=sigma_s)

#     # Calculate gradients of the input noisy image
#     gradient_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)

#     # Calculate gradients of the smoothed image
#     smooth_gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
#     smooth_gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
#     cv2.imshow('Original Image', input_image)
#     cv2.imshow('smoothed_image', smoothed_image)
#     cv2.imshow('gradient_x', smooth_gradient_x)
#     cv2.imshow('gradient_y', smooth_gradient_y)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # Initialize the output image
#     u = np.copy(smoothed_image)

#     # # Iterate over the image pixels
#     for i in range(input_image.shape[0]):
#         for j in range(input_image.shape[1]):
#             # Calculate fBLF(∇g[q])
#            fBLF_x = smooth_gradient_x[i, j]
#            fBLF_y = smooth_gradient_y[i, j]

#     #         # Calculate the weights
#     #         weights = np.exp(-(np.abs(gradient_x[i, j] - gradient_x) + np.abs(gradient_y[i, j] - gradient_y)) / sigma_r)
#     #         weights /= np.sum(weights)

#     #         # Update the output image pixel
#     #         u[i, j] = input_image[i, j] + lambda_value * (
#     #             (smooth_gradient_x[i, j] - fBLF_x) ** 2 + (smooth_gradient_y[i, j] - fBLF_y) ** 2
#     #         ) * np.sum(weights)

#     return u

# # Load the noisy infrared image
# input_image = cv2.imread('../../Data/need_label/PCV/4/03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)

# # Apply the BLF-LS algorithm
# result_image = blf_ls(input_image, lambda_value=1024)

# # Save the result image
# #cv2.imwrite('result_image.png', result_image)

# # Display the result image and original image
# cv2.imshow('Result Image', result_image)
# cv2.imshow('Original Image', input_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 读取 OCTA 图像
# octa_image = cv2.imread('../../Data/images_info/INPAINT_TELEA03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)


import cv2
import numpy as np
from scipy import fftpack
lambda_value = 1024
# OCTA Stripe Noise Removal using bilateral filter
def bilateral_filter(input_image, sigma_s=10, sigma_r=50):
    # Bilateral filter the input image
    smoothed_image = cv2.bilateralFilter(input_image, d=0, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    return smoothed_image

# OCTA Stripe Noise Removal 
def octa_stripe_noise_removal(input_image, sigma_s=10, sigma_r=50):
    # Bilateral filter the input image
    smoothed_image = bilateral_filter(input_image, sigma_s, sigma_r)

    # Calculate gradients of the input noisy image
    gradient_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradients of the smoothed image
    smooth_gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    smooth_gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

    # Initialize the output image
    u = np.copy(smoothed_image)

    # Iterate over the image pixels
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            # Calculate fBLF(∇g[q])
            fBLF_x = smooth_gradient_x[i, j]
            fBLF_y = smooth_gradient_y[i, j]

            # Calculate the weights
            weights = np.exp(-(np.abs(gradient_x[i, j] - gradient_x) + np.abs(gradient_y[i, j] - gradient_y)) / sigma_r)
            weights /= np.sum(weights)

            # Update the output image pixel
            u[i, j] = input_image[i, j] + lambda_value * (
                (smooth_gradient_x[i, j] - fBLF_x) ** 2 + (smooth_gradient_y[i, j] - fBLF_y) ** 2
            ) * np.sum(weights)

    return u


# Load the noisy infrared image
input_image = cv2.imread('../../Data/images_info/INPAINT_TELEA03959415_R_20220714.png', cv2.IMREAD_GRAYSCALE)

# Apply the BLF-LS algorithm
result_image = octa_stripe_noise_removal(input_image, sigma_s=2, sigma_r=0.04)

sub = cv2.subtract(input_image, result_image)

# Save the result image
#cv2.imwrite('result_image.png', result_image)

# Display the result image and original image
cv2.imshow('Result Image', result_image)
cv2.imshow('sub', sub)
cv2.imshow('Original Image', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


