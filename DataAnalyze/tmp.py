import pathlib
import os
import cv2
import shutil
import tools.tools as tools
path = "../../Data/Label"
labeled = path + '/' + 'labeled'
os.makedirs(labeled, exist_ok=True)

for image_path in pathlib.Path(path).iterdir():
    for date in pathlib.Path(image_path).iterdir():
        for eye in  pathlib.Path(date).iterdir():
            # if is a folder
            if os.path.isdir(eye):
                file = sorted(pathlib.Path(eye).iterdir())
                # show CNV not in file name
                check = False
                for i in range(len(file)):
                    if 'CNV' in file[i].name:
                        check = True
                        img = cv2.imread(str(file[i]))
                        img = img * 255
                        img = img.astype('uint8')
                        route = os.path.join(labeled, image_path.name, date.name, eye.name)
                        os.makedirs(route, exist_ok=True)
                        print(route)
                        if 'CC' in file[i].name:
                            cv2.imwrite(os.path.join(route, 'label_4.png'), img)
                        else:
                            cv2.imwrite(os.path.join(route, 'label_3.png'), img )
                        # cv2.imwrite(labeled + image_path.name + date.name + eye.name + '.png', img)
                if not check:
                    print(eye)
                    

                
                         
                        