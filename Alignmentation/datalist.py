import pathlib as pl
import json

# Path: datalist.py
cases = dict()
for patient in pl.Path('../../Data/OCTA').iterdir():
    cases[patient.name] = dict()
    for date in patient.iterdir():
        cases[patient.name][date.name] = list()
        for eye in date.iterdir():
            if eye.name !="L" and eye.name !="R":
                print(patient.name, date.name)
            cases[patient.name][date.name].append(eye.name)

print('patient',len(cases))
json_data = json.dumps(cases, indent=4)
with open("cases.json", "w") as outfile:
    outfile.write(json_data)

file = open('cases.txt', 'w')
for patient in cases:
    file.write(patient + '\n')
    for date in cases[patient]:
        if "L" in cases[patient][date] and "R" in cases[patient][date]:
            file.write('date: '+ date + ' L R' + '\n')
        elif "L" in cases[patient][date]:
            file.write('date: '+ date + ' L' + '\n')
        elif "R" in cases[patient][date]:
            file.write('date: '+ date + ' R' + '\n')
    file.write('==============================\n')

file.close()
# pip install aspose-words
# txt to pdf
import aspose.words as aw
txt_name_list = ['cases.txt','other.txt']
for txt_name in txt_name_list:
    txt = aw.Document(txt_name)
    txt.save(txt_name[:-4] + '.pdf', aw.SaveFormat.PDF)

