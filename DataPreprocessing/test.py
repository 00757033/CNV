import numpy as np
import matplotlib.pyplot as plt
import cv2 


# Load the octa image
image = cv2.imread("..\\..\\Data\\PCV_0205\\ALL\\1\\10284504_L_20180108.png", cv2.IMREAD_GRAYSCALE)

# # Binarize the image
# _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Calculate the box counting fractal dimension
# def box_count(image, box_size):
#     count = 0
#     h, w = image.shape
#     for y in range(0, h, box_size):
#         for x in range(0, w, box_size):
#             if np.any(image[y:y+box_size, x:x+box_size] == 255):
#                 count += 1
#     return count

# box_sizes = range(1, 304//2)
# counts = [box_count(binary_image, box_size) for box_size in box_sizes]
# fractal_dimension = -np.polyfit(np.log(box_sizes), np.log(counts), 1)[0]

# # Find the best box size
# best_box_size = box_sizes[np.argmax(np.diff(np.log(counts)))]
# print("Fractal Dimension:", fractal_dimension)
# print("Best Box Size:", best_box_size)

# # Display the box in the image
# plt.imshow(binary_image, cmap='gray')
# plt.xticks([]), plt.yticks([])

# for y in range(0, image.shape[0], best_box_size):
#     plt.axhline(y=y, color='r', linestyle=':')
# for x in range(0, image.shape[1], best_box_size):
#     plt.axvline(x=x, color='r', linestyle=':')

# plt.show()


# Threshold the image to create a binary image
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Implement the box counting algorithm
def box_count(binary_image, box_size):
    count = 0
    height, width = binary_image.shape
    for i in range(0, height - box_size + 1, box_size):
        for j in range(0, width - box_size + 1, box_size):
            if np.max(binary_image[i:i+box_size, j:j+box_size]) > 0:
                count += 1
    return count

# Calculate the number of boxes for different box sizes
box_sizes = range(1, min(binary_image.shape))  # Box sizes from 1 to half of image size
box_counts = [box_count(binary_image, box_size) for box_size in box_sizes]

# Plot number of boxes versus box size
plt.figure()
plt.plot(box_sizes, box_counts, 'bo-')
plt.title('Box Counting Fractal Dimension')
plt.xlabel('Box Size')
plt.ylabel('Number of Boxes')
plt.grid(True)

# Find the best box size
best_box_size_index = np.argmax(np.gradient(np.log(box_counts), np.log(box_sizes)))
best_box_size = box_sizes[best_box_size_index]
plt.plot(best_box_size, box_counts[best_box_size_index], 'ro')  # Highlight the best box size
plt.text(best_box_size, box_counts[best_box_size_index], f'Best Box Size: {best_box_size}', verticalalignment='bottom')

plt.show()

# Calculate the fractal dimension
fractal_dimension = -np.gradient(np.log(box_counts), np.log(box_sizes))[best_box_size_index]
print("Fractal Dimension:", fractal_dimension)

# # Display the box in the image
plt.imshow(binary_image, cmap='gray')
plt.xticks([]), plt.yticks([])
for y in range(0, image.shape[0], best_box_size):
    plt.axhline(y=y, color='r', linestyle=':')
for x in range(0, image.shape[1], best_box_size):
    plt.axvline(x=x, color='r', linestyle=':')

plt.show()

# # Binarize the image
# _, binary_image = cv2.threshold(octa_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# # Calculate the box counting fractal dimension
# def box_count(image, box_size):
#     h, w = image.shape
#     count = 0
#     for y in range(0, h, box_size):
#         for x in range(0, w, box_size):
#             if np.max(image[y:y+box_size, x:x+box_size]) > 0:
#                 count += 1
#     return count

# box_sizes = np.arange(1,304//2, 2)
# counts = []

# for box_size in box_sizes:
#     counts.append(box_count(binary_image, box_size))
    
# # Fit a line to the points and calculate the fractal dimension
# coeffs = np.polyfit(np.log(1/box_sizes), np.log(counts), 1)
# fractal_dimension = -coeffs[0]

# # Plot the box counting data
# plt.figure()
# plt.scatter(np.log(1/box_sizes), np.log(counts))
# plt.plot(np.log(1/box_sizes), np.polyval(coeffs, np.log(1/box_sizes)), 'r--')
# plt.xlabel('log(1/box size)')
# plt.ylabel('log(count)')
# plt.title(f'Box Counting Fractal Dimension: {fractal_dimension:.2f}')
# plt.show()
