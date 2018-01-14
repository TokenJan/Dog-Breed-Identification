import os

for folder in os.listdir('../input/Images'):
    changed_folder = folder[10:].lower()
    os.rename('../input/Images/' + folder, '../input/Images/' + changed_folder)

os.rename('../input/train', '../input/train_old')
os.rename('../input/Images', '../input/train')