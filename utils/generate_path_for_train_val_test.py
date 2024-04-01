import os


data_dir = os.path.join('/home/wanchao/new_drive/sk2fabrication/data/sketch_out/open_house/256x256')
file_list = os.listdir(data_dir)

train_val_test_folder_path='/home/wanchao/new_drive/sk2fabrication/data/sketch_out/train_val_test/open_house'
train_txt=os.path.join(train_val_test_folder_path, 'train.txt')

with open(train_txt, 'w+') as f:
    for path in file_list:
        f.write("{}\n".format(path))
print()