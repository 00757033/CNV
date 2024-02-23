import numpy as np
from scipy.ndimage import convolve
from scipy.signal import convolve2d
import cv2

# def bilateral_filter(img, sigma_s, sigma_r):
#     """
#     Apply bilateral filter to an input image.
    
#     Args:
#     img (numpy.ndarray): Input image.
#     sigma_s (float): Geometric proximity factor.
#     sigma_r (float): Intensity similarity factor.
    
#     Returns:
#     filtered_img (numpy.ndarray): Filtered image.
#     """
#     # Create a mesh grid of pixel coordinates
#     x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
#     x_diff = x - x.T
#     y_diff = y - y.T
    
#     # Calculate the spatial kernel using geometric proximity factor (sigma_s)
#     spatial_kernel = np.exp(-(x_diff**2 + y_diff**2) / (2 * sigma_s**2))
    
#     # Calculate the intensity kernel using intensity similarity factor (sigma_r)
#     intensity_kernel = np.exp(-(img - img.T)**2 / (2 * sigma_r**2))
    
#     # Compute the final bilateral filter kernel
#     kernel = spatial_kernel * intensity_kernel
    
#     # Normalize the kernel
#     kernel /= np.sum(kernel, axis=1, keepdims=True)
    
#     # Apply the bilateral filter to the input image
#     filtered_img = np.matmul(kernel, img)
    
#     return filtered_img

# import numpy as np

# def BLF_LS(g, λ, σs, σr):
#     """
#     Apply BLF-LS (Bilateral Filter Least Squares) smoothing to an input noisy infrared image.
    
#     Args:
#     g (numpy.ndarray): Input noisy infrared image.
#     λ (float): Smoothing parameter that controls the balance between two parts of the model.
#     σs (float): Geometric proximity factor for bilateral filter.
#     σr (float): Intensity similarity factor for bilateral filter.
    
#     Returns:
#     u (numpy.ndarray): Smoothed output image.
#     """
#     # Apply bilateral filtering to the image gradient (∇g[q]) along the x-axis and y-axis
#     smoothed_gradient_x = bilateral_filter(np.gradient(g, axis=1), σs, σr)
#     smoothed_gradient_y = bilateral_filter(np.gradient(g, axis=0), σs, σr)
#     # show the smoothed_gradient
#     cv2.imshow('smoothed_gradient_x',smoothed_gradient_x)
#     cv2.imshow('smoothed_gradient_y',smoothed_gradient_y)


#     # Initialize the smoothed output image u
#     u = cv2.bilateralFilter(g, 5, 0.1, 0.1)

#     # frequency domain
#     u = np.fft.fft2(u)


#     # turn back to spatial domain
#     u = np.fft.ifft2(u)
#     u = np.real(u)
#     u = np.uint8(u)
#     print(u)

#     # # display the smoothed_gradient
#     cv2.imshow('u',u)

#     cv2.waitKey(0)


    
#     # Iterate over each pixel and perform least squares smoothing
#     for i in range(g.shape[0]):
#         for j in range(g.shape[1]):
#             diff_x = smoothed_gradient_x[i, j] - smoothed_gradient_x
#             diff_y = smoothed_gradient_y[i, j] - smoothed_gradient_y
#             # bilateral_filter
#             diff_x = bilateral_filter(diff_x, σs, σr)
#             diff_y = bilateral_filter(diff_y, σs, σr)

#             squared_diff = (u[i, j] - g)**2 + λ * (diff_x**2 + diff_y**2)
            
#             # Use np.nan_to_num to replace NaN and Inf values with a specified value (e.g., 0)
#             squared_diff = np.nan_to_num(squared_diff)
            
#             # Sum the squared differences
#             squared_diff_sum = np.sum(squared_diff)

#             # Compute the smoothed output image u
#             u[i, j] = squared_diff_sum / (g.shape[0] * g.shape[1])
#             print(f'({i}, {j}) {u[i, j]}')

#     return u

def bilateral_filter_least_squares(input_image, lambda_value=1024, sigma_s=2, sigma_r=0.07):
    # Convert input image to grayscale if it's not already
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Calculate gradient of the input image
    gradient_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=5)
    intensity_similarity = 0.0
    smoothed_image = np.copy(input_image)
    # Declare and initialize intensity_similarity before the loop
    intensity_similarity = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.float64)
    input_image = input_image.astype(np.float32)
    # Declare and initialize intensity_similarity before the loops
    intensity_similarity_x = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.float64)
    intensity_similarity_y = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.float64)

    for p in range(1, input_image.shape[0] - 1):
        for q in range(1, input_image.shape[1] - 1):
            Zm = 0
            weighted_sum = 0

            # Calculate intensity_similarity outside of the innermost loop
            for m in range(p - 1, p + 2):
                for n in range(q - 1, q + 2):
                    spatial_distance = np.linalg.norm([abs(m - p), abs(n - q)])
                    
                    # Calculate intensity_similarity for both x and y components
                    intensity_similarity_x[p, q] = np.abs(gradient_x[p, q] - gradient_x[m, n])
                    intensity_similarity_y[p, q] = np.abs(gradient_y[p, q] - gradient_y[m, n])

                    # Clip the intensity_similarity to a reasonable range
                    max_intensity_difference = 100.0
                    intensity_similarity_x[p, q] = np.clip(intensity_similarity_x[p, q], -max_intensity_difference, max_intensity_difference)
                    intensity_similarity_y[p, q] = np.clip(intensity_similarity_y[p, q], -max_intensity_difference, max_intensity_difference)

                    weight = np.exp(-spatial_distance / (2 * sigma_s ** 2)) * \
                            np.exp(-intensity_similarity_x[p, q] / (2 * sigma_r ** 2)) * \
                            np.exp(-intensity_similarity_y[p, q] / (2 * sigma_r ** 2))

                    if np.isfinite(weight) and np.isfinite(gradient_x[n, q]):
                        Zm += weight
                        weighted_sum += weight * gradient_x[n, q]

            if Zm != 0:  # Ensure Zm is not zero to avoid division by zero
                smoothed_gradient_x = weighted_sum / Zm
            else:
                smoothed_gradient_x = 0  # Assign a default value or handle this case as needed

            # Update the smoothed image gradient in the x-direction
            smoothed_image[p, q] = smoothed_gradient_x


    return smoothed_image



def guided_filter_1d(guidance, input_image, radius, epsilon=1e-8):
    """
    Apply a 1D column gradient domain guided image filter.

    Args:
    guidance (numpy.ndarray): Guidance image.
    input_image (numpy.ndarray): Input image to be filtered.
    radius (int): Radius of the square neighborhood.
    epsilon (float): A small positive constant to avoid division by zero.

    Returns:
    filtered_image (numpy.ndarray): Filtered image.
    """

    # Compute the gradients of the guidance image
    dI_x = np.gradient(guidance, axis=0)
    
    # Initialize the filtered image
    filtered_image = np.zeros_like(input_image)
    
    height, width = input_image.shape

    for x in range(width):
        # Define the column range for the neighborhood
        x_start = max(0, x - radius)
        x_end = min(width - 1, x + radius)
        
        # Extract the current column from the guidance image
        I_x = guidance[:, x_start:x_end + 1]
        
        # Extract the current column from the input image
        p_x = input_image[:, x_start:x_end + 1]

        # Compute the local means and variances
        mean_I_x = np.mean(I_x, axis=1, keepdims=True)
        mean_p_x = np.mean(p_x, axis=1, keepdims=True)
        cov_Ip_x = np.mean(I_x * p_x, axis=1, keepdims=True) - mean_I_x * mean_p_x
        var_I_x = np.mean(I_x * I_x, axis=1, keepdims=True) - mean_I_x * mean_I_x

        # Compute the filter coefficients
        a_x = cov_Ip_x / (var_I_x + epsilon)
        b_x = mean_p_x - a_x * mean_I_x

        # Compute the output pixel values for the current column
        filtered_image[:, x] = np.sum(a_x * p_x + b_x, axis=1)

    return filtered_image


# Example usage
input_noisy_infrared_image = cv2.imread("C:\\Users\\asus\\Downloads\\images.jpg",cv2.IMREAD_GRAYSCALE)
λ = 1024
σs = 1  # Adjust these values as needed
σr = 10
input_noisy_infrared_image = cv2.resize(input_noisy_infrared_image, (152, 152))

# Apply BLF-LS smoothing to the input noisy infrared image
output_smoothed_image = bilateral_filter_least_squares(input_noisy_infrared_image, λ, σs, σr)

# subtract the smoothed image from the input image
sub = input_noisy_infrared_image - output_smoothed_image

# Display the image
cv2.imshow('input_noisy_infrared_image',input_noisy_infrared_image)
cv2.imshow('output_smoothed_image',output_smoothed_image)
cv2.imshow('sub',sub)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Set the filter radius and epsilon
radius = 5
epsilon = 1e-6

# Apply the guided filter
filtered_image = guided_filter_1d(input_noisy_infrared_image, sub, radius, epsilon)

# Display the filtered image
cv2.imshow('filtered_image',filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




    



    
