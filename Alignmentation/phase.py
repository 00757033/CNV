import cv2
import numpy as np
import matplotlib.pyplot as plt

def phase_correlation(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Apply Fourier Transform
    f1 = np.fft.fft2(gray1)
    f2 = np.fft.fft2(gray2)

    # Calculate cross power spectrum
    cross_power_spectrum = np.multiply(f1, np.conj(f2))

    # Calculate phase correlation
    phase_correlation = np.fft.ifft2(cross_power_spectrum)
    phase_correlation = np.fft.fftshift(phase_correlation)

    # Calculate magnitude and angle
    magnitude = np.abs(phase_correlation)
    angle = np.angle(phase_correlation)

    # Find the peak in the magnitude
    _, _, _, max_loc = cv2.minMaxLoc(magnitude)

    # Calculate translation (shift)
    rows, cols = gray2.shape
    center = (cols // 2, rows // 2)
    shift = (max_loc[0] - center[0], max_loc[1] - center[1])

    return shift   

def alignment( image1, image2):
    img1 = cv2.imread(pre_treatment)
    img2 = cv2.imread(post_treatment)
    shift = phase_correlation(img1, img2)
    print(shift)
    # 對位
    rows, cols,_ = img2.shape
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    result = cv2.warpAffine(img2, M, (cols, rows))
    result[result == 0] = img1[result == 0]
    plt.imshow(result)
    plt.show()



if __name__ == '__main__':
    pre_treatment = "..\\..\\Data\\PCV_1017\\ALL\\1\\03408585_R_20210422.png"
    post_treatment = "..\\..\\Data\\PCV_1017\\ALL\\1\\03408585_R_20210929.png"
    alignment(pre_treatment, post_treatment)



    # plt.ims





    




