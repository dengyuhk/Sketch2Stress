import os

img_apth='/home/wanchao/new_drive/sk2fabrication/data/sketch_out/shapeAirplane/256x256'
source_path='/home/wanchao/new_drive/sk2fabrication/data/sketch_out/train_val_test/shapeAirplane'
train_txt='train.txt'
test_txt='test.txt'

# target_data_path='/media/deng/58E66EC565EA1C09/sk2fabricaion/data'
file_list=os.listdir(img_apth)


for txt in [train_txt,test_txt]:
    train_val_test_txt_path= os.path.join(source_path,txt)
    with open(train_val_test_txt_path, 'r') as fh:
        imgs_all = fh.readlines()

    mesh_list=[ele.split('-')[0] for ele in imgs_all]
    mesh_list=list(set(mesh_list))

    new_paths= []
    for mesh_id in range(len(mesh_list)):
        print('{}/{}'.format(mesh_id,len(mesh_list)))
        mesh_name = mesh_list[mesh_id]
        new_path = [ele for ele in file_list if ele.split('-')[0] == mesh_name]
        new_paths = new_paths + new_path


    with open(train_val_test_txt_path.split('.txt')[0]+'new.txt', 'w+') as f:
        for path in new_paths:
            f.write("{}\n".format(path))
