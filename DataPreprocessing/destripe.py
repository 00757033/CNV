
import numpy as np
import cv2
import pathlib as pl
import matplotlib.pyplot as plt

def destripe_octa_image(octa_image):
        # Convert OCTA image to floating point for processing
    # smooth
    # octa_image = cv2.bilateralFilter(octa_image, d=0, sigmaColor=50, sigmaSpace=10)
    # octa_image = 
    octa_image = octa_image.astype(np.float32)

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
    destriped_image = cv2.normalize(destriped_image, None, 0, 255, cv2.NORM_MINMAX)
    print(max(destriped_image.ravel()))
    # Convert the destriped image back to uint8 for display
    destriped_image = destriped_image.astype(np.uint8)

    # add weighted
    octa_image = octa_image.astype(destriped_image.dtype)

    destriped_image = cv2.addWeighted(octa_image, 0.75, destriped_image, 0.25, 0)

    # # Convert OCTA image to floating point for processing
    # octa_image = octa_image.astype(np.float32)

    # mean = np.mean(octa_image)
    # # Calculate the mean intensity profile along the A-scan (vertical) direction
    # mean_profile_1 = np.mean(octa_image, axis=1)
    # mean_profile_1[mean_profile_1 < mean] = 0.0
    # destriped_image = octa_image.copy()
    # # 刪除條紋並補差值
     
    # # print(mean_profile_1)
    # # destriped_image = destriped_image.astype(np.uint8)
    # # for i in range(mean_profile_1.shape[0]):
    # #     # Subtract the image
    # #     destriped_image[i, :] -= mean_profile_1[i]

    # #     inpaintRadius = 3
    # #     mask = destriped_image < 5
    # #     mask = mask.astype(np.uint8)
    # #     destriped_image = destriped_image.astype(np.uint8)
    # #     destriped_image = cv2.inpaint(destriped_image, mask, inpaintRadius, cv2.INPAINT_TELEA)


    # # Calculate the mean intensity profile along the B-scan (horizontal) direction
    # mean_profile_2 = np.mean(octa_image, axis=0)
    # mean_profile_2[mean_profile_2 < mean] = 0


    # for i in range(octa_image.shape[1]):
    #     destriped_image[:, i] -= mean_profile_2[i]

    #     inpaintRadius = 3
    #     mask = destriped_image < 5
    #     mask = mask.astype(np.uint8)
    #     destriped_image = destriped_image.astype(np.uint8)
    #     destriped_image = cv2.inpaint(destriped_image, mask, inpaintRadius, cv2.INPAINT_TELEA)

    # # Subtract the mean profile from the entire image to remove vertical stripes
    # # destriped_image -= mean_profile_1[:, np.newaxis]


    # # Subtract the mean profile from the entire image to remove horizontal stripes
    # # destriped_image -= mean_profile_2[np.newaxis, :]

    # # Clip negative values to ensure the image remains non-negative
    # destriped_image[destriped_image < 0] = 0

    # mask = destriped_image < 5
    # mask = mask.astype(np.uint8)
    # destriped_image = destriped_image.astype(np.uint8)

    


    # remove_img = cv2.cvtColor(destriped_image, cv2.COLOR_RGB2BGR)
    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    #dst = cv2.inpaint(destriped_image, mask, 3, cv2.INPAINT_TELEA)
    # Return the destriped image
    return destriped_image



if __name__ == "__main__":
    # Load the OCTA image
    for image in pl.Path("..\..\Data\PCV_0915\OR\images").iterdir():
        image_name = image.name
        image_path = str(image)
        print(image_name)
        octa_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # octa_image = cv2.imread("..\..\Data\PCV_0915\CC\images\CC_10713526_L_20211129.png", cv2.IMREAD_GRAYSCALE)
        # octa_image = cv2.resize(octa_image, (100, 100))
        # Apply destriping
        destriped_octa  = destripe_octa_image(octa_image)

        # # PSNR
        # psnr = cv2.PSNR(octa_image, destriped_octa)
        # print(psnr)

        # Detect potential missing or noisy areas (adjust the threshold as needed)

        # Use cv2.inpaint to fill the detected areas
        # inpainted_image = cv2.inpaint(destriped_octa, mask.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Display the original OCTA image, destriped image, and inpainted image
        # octa_image = cv2.resize(octa_image, (152, 152))
        # destriped_octa = cv2.resize(destriped_octa, (152, 152))
        p2 = cv2.addWeighted(octa_image, 0.25, destriped_octa, 0.5, 0)
        p4 = cv2.addWeighted(octa_image, 0.5, destriped_octa, 0.75, 0)

        destriped_octa = cv2.normalize(destriped_octa, None, 0, 255, cv2.NORM_MINMAX)
        p2 = cv2.normalize(p2, None, 0, 255, cv2.NORM_MINMAX)
        p4 = cv2.normalize(p4, None, 0, 255, cv2.NORM_MINMAX)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # destriped_octa = clahe.apply(destriped_octa)
        # p2 = clahe.apply(p2)
        # p4 = clahe.apply(p4)



        plt.figure(figsize=(15, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(octa_image, cmap='gray')
        plt.title('Original OCTA Image')
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(destriped_octa, cmap='gray')
        plt.title('Destriped OCTA Image')
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(p2, cmap='gray')
        plt.title('25% Original + 75% Destriped')
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(p4, cmap='gray')
        plt.title('50% Original + 50% Destriped')
        plt.axis('off')

        plt.show()
        

