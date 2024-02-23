import pathlib as pl
import json


file = "C:\\Users\\CcepWang\\Desktop\\sharon\\研究\\Data\\PCV_0205\\ALL\\1\\"
name = set()
for data in pl.Path(file).iterdir():
    data_name = data.name.split('_')[0]
    name.add(data_name)

name = sorted(name)    
# save as txt
file = open('cases.txt', 'w')
for patient in name:
    file.write(patient + '\n')
file.close()

    