import os
import pathlib as pl

# check case by group
def count_case(path,layers = ['CC','OR']):
    for layer in layers:
        name = '_'+layer
        print(layer)
        for dir in pl.Path(path +name).iterdir():
            dir_name = dir.name
            print(dir_name)
            group = dict()
            for files in pl.Path(dir / 'images').iterdir():
                file_split = files.name.split('_')
                file_group = file_split[1] + '_' + file_split[2]
                if file_group not in group:
                    group[file_group] = 1
                else:
                    group[file_group] += 1
            count = dict()
            for key,value in group.items():
                if value not in count:
                    count[value] = 1
                else:
                    count[value] += 1
            count = sorted(count.items(), key=lambda x: x[0])
            print(count)





if __name__ == "__main__":
    path = "../../Data"
    date = '0819'
    disease = 'PCV'
    path_base =  path + "/" + disease + "_"+ date
    image_path  = path + "/" + "PCV_0819" + '/' + 'trainset' + '/' + 'otsu'

    count_case(image_path)
