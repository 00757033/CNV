import cv2
import numpy as np

def blf_ls(input_image, lambda_value=1024, sigma_s=1.0, sigma_r=0.1):
    # Convert the input image to float32
    input_image = np.float32(input_image)

    # Apply bilateral filtering to smooth the image gradient
    smoothed_gradient_x = cv2.bilateralFilter(input_image, -1, sigma_s, sigma_r)
    smoothed_gradient_y = cv2.bilateralFilter(input_image, -1, sigma_s, sigma_r)

    # Calculate the gradient of the smoothed gradient image
    gradient_x = np.gradient(smoothed_gradient_x, axis=1)
    gradient_y = np.gradient(smoothed_gradient_y, axis=0)
    cv2.imshow('gradient_x', gradient_x)
    cv2.imshow('gradient_y', gradient_y)

    gradient_x = np.float32(gradient_x)
    gradient_y = np.float32(gradient_y)


    # Initialize the output image u
    u = np.copy(input_image)
    # FFT and IFFT
    u = np.fft.fft2(u)
    u = np.fft.ifft2(u)
    u = np.real(u)
    u = np.uint8(u) 
    print(u)

    # Define the neighborhood window
    neighborhood_size = 3
    half_window = neighborhood_size // 2

    for i in range(half_window, u.shape[0] - half_window):
        for j in range(half_window, u.shape[1] - half_window):
            # Extract the local neighborhood
            neighborhood_x = gradient_x[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
            neighborhood_y = gradient_y[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]

            # Compute Zm for normalization
            Zm = np.sum(np.exp(-np.sqrt(np.square(neighborhood_x) + np.square(neighborhood_y)) / sigma_s ** 2))

            # Calculate the weighted sum
            weighted_sum_x = np.sum(neighborhood_x * np.exp(-np.sqrt(np.square(neighborhood_x) + np.square(neighborhood_y)) / sigma_s ** 2))
            weighted_sum_y = np.sum(neighborhood_y * np.exp(-np.sqrt(np.square(neighborhood_x) + np.square(neighborhood_y)) / sigma_s ** 2))

            # Update the output image u
            u[i, j] = input_image[i, j] + lambda_value * (gradient_x[i, j] - weighted_sum_x / Zm) ** 2 + \
                      lambda_value * (gradient_y[i, j] - weighted_sum_y / Zm) ** 2

    return u


# Load your input noisy infrared image using cv2.imread
input_image = cv2.imread("C:\\Users\\asus\\Downloads\\images.jpg", cv2.IMREAD_GRAYSCALE)
# input_image = cv2.resize(input_image, (150, 150))

# Apply the BLF-LS smoothing algorithm
output_image = blf_ls(input_image, lambda_value=1024, sigma_s=250, sigma_r=100)
sub = cv2.subtract(input_image, output_image)
# Save the output image using cv2.imwrite
# cv2.imwrite('output_image.png', output_image)

# Display the input and output images using cv2.imshow
cv2.imshow('input_image', input_image)
cv2.imshow('output_image', output_image)
cv2.imshow('sub', sub)
cv2.waitKey(0)
