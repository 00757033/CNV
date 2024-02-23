import contours
import cv2
import numpy as np
import os
import pathlib as pl

def relabeler(path):
    for patientID in pl.Path(path).iterdir():
        for date in patientID.iterdir():
            for OD_OS in date.iterdir():
                for files in OD_OS.iterdir():
                    if (files.stem).startswith('label_'):
                        mask_path = str(files)
                        image_path = str(files).replace('new_label','OCTA')
                        image_path = image_path.replace('label_','')
                        image = cv2.imread(image_path, 0)

                        mask = cv2.imread(mask_path, 0)
                        image_mask = image.copy()
                        image_mask[mask == 0] = 0
                        # detect blood vessels
                        denoised_image,binary_image,img = contours.detect_blood_vessels(image_mask)
                        cv2.imshow("img", img)
                        cv2.waitKey(5)

                        # cv2.imwrite(mask_path, img)
                        # print(mask_path)

if __name__ == "__main__":
    relabeler('..\\..\\Data\\new_label')
