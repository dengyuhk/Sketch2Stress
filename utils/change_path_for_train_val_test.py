import os


source_path='/home/wanchao/new_drive/sk2fabrication/data/sketch_out/train_val_test/psbAirplane'
train_txt='train.txt'
val_txt='val.txt'
test_txt='test.txt'

# target_data_path='/media/deng/58E66EC565EA1C09/sk2fabricaion/data'

for txt in [train_txt,val_txt,test_txt]:
    train_val_test_txt_path= os.path.join(source_path,txt)
    with open(train_val_test_txt_path, 'r') as fh:
        imgs_all = fh.readlines()

    imgs_all_new=[ele.split('256x256/')[-1]  for ele in imgs_all]
    with open(train_val_test_txt_path, 'w+') as f:
        for path in imgs_all_new:
            f.write("{}".format(path))
