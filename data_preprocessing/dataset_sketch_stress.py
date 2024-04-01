##create by deng
import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import h5py
import matplotlib.pyplot as plt
import cv2


class DatasetSkStress(Dataset):
    def __init__(self,_dataPath,_cate,_normalized_scale,_phase='train',disorderKey=False):
        'Initialization'
        self.cate = _cate
        self.phase= _phase
        self._normalized_scale=_normalized_scale
        self.dataPath=_dataPath
        datasets =['ShapeNet','PSB','ShapeSeg','AniHead','OpenSketch']
        cates = {
                 'chair': 'psbChair',
                 'airplane':'psbAirplane',
                 # 'table': {'PartFPN': 'Table', 'PartFPN_Specified_Part': 'Table'},
                 # 'car':{'PartFPN': 'Car', 'PartFPN_Specified_Part': 'Car'},
                 # 'guitar':{'PartFPN': 'Guitar', 'PartFPN_Specified_Part': 'Guitar'},
                 # 'monitor':{'PartFPN': 'Monitor', 'PartFPN_Specified_Part': 'Monitor'},
                 # 'lampa':{'PartFPN': 'LampA', 'PartFPN_Specified_Part': 'LampA'},
                 # 'vase': {'PartFPN': 'psbVase', 'PartFPN': 'psbVase'},
                 # 'mug':   {'PartFPN': 'Mug', 'PartFPN_Specified_Part': 'Mug'},
                 # 'lampc': {'PartFPN': 'LampC', 'PartFPN_Specified_Part': 'LampC'},
                 }

        self.cate_name = cates[self.cate]
        self.train_val_dir = osp.join(self.dataPath,'train_val_test')
        self.training_cate_path = os.path.join( self.train_val_dir, self.cate_name)


        self.hdf5_dir_sketch = osp.join(self.dataPath, '{}_{}.hdf5'.format(self.cate_name,'sketch'))
        self.hdf5_dir_stress = osp.join(self.dataPath, '{}_{}.hdf5'.format(self.cate_name, 'stress'))

        if len(os.listdir(self.training_cate_path))==0:
            self._processed_data = self._process_datasets_paired_sk_stress()
        else:
            pass

    def _process_datasets_paired_sk_stress(self):
        training_path= os.path.join(self.dataPath,'train_val_test')
        if not os.path.exists(training_path): os.makedirs(training_path)
        training_cate_path=  os.path.join(training_path,self.cate_name)
        if not os.path.exists(training_cate_path): os.makedirs(training_cate_path)

        dataset_sketch=h5py.File(self.hdf5_dir_sketch, 'r')
        dateset_stress=h5py.File(self.hdf5_dir_stress, 'r')

        raw_data_sketch = dataset_sketch['sketch']
        # raw_data_normal=raw_data_sketch['normal']
        # raw_data_depth =raw_data_sketch['depth']
        raw_data_f_node= dataset_sketch['pixel']
        raw_data_stress = dateset_stress['stress']

        sample_count=0
        for _index in list(raw_data_sketch.keys()):
            print('[{}/{}th sketch has been processing ..... ]'.format(sample_count, len(list(raw_data_sketch.keys()))))
            sample_count = sample_count + 1

            ## sketch information
            sketch_mesh_name,sketch_azim_degree,skecth_elev_degree,sketch_view_count  = raw_data_sketch[_index].attrs['mesh_filename'],raw_data_sketch[_index].attrs['azim_degree'],raw_data_sketch[_index].attrs['elev_degree'],raw_data_sketch[_index].attrs['view_count']
            sketch_info= [sketch_mesh_name,sketch_azim_degree,skecth_elev_degree,sketch_view_count]

            sketch_gt=np.array(raw_data_sketch[_index])
            force_node_gts= np.array(raw_data_f_node[_index])
            stress_gts=np.array(raw_data_stress[_index])
            self.save_paired_sk_force_stress(sketch_gt,force_node_gts,stress_gts,sketch_info,training_cate_path)

    def save_paired_sk_force_stress(self,sketch,force_nodes,stresses,sketch_info,training_cate_path):
        _sk_path = '{}-{}-{}-{}-{}.png'.format(training_cate_path+'/'+sketch_info[0].split('.obj')[0],  #
                                                       int(sketch_info[1]),
                                                       int(sketch_info[2]),
                                                       'view' + str(sketch_info[3]),
                                                       'sketch')
        cv2.imwrite(_sk_path,sketch*255)

        for force_index in range(len(force_nodes)):
            force_mask = np.zeros((256, 256))
            row, col = force_nodes[force_index]
            region_size = 2
            force_mask[row - region_size:row + region_size, col - region_size:col + region_size] = 1
            stress_map= stresses[force_index]

            ## save to the harddrive
            _force_node_mask_path = '{}-{}-{}-{}-{}-{}.png'.format(training_cate_path + '/' + sketch_info[0].split('.obj')[0],  #
                                                   int(sketch_info[1]),
                                                   int(sketch_info[2]),
                                                   'view' + str(sketch_info[3]),
                                                   'sketch','['+str(row)+'_'+str(col)+']')
            _stress_path='{}-{}-{}-{}-{}-{}-{}.png'.format(training_cate_path + '/' + sketch_info[0].split('.obj')[0],  #
                                                   int(sketch_info[1]),
                                                   int(sketch_info[2]),
                                                   'view' + str(sketch_info[3]),
                                                   'sketch','['+str(row)+'_'+str(col)+']','stress')

            cv2.imwrite(_force_node_mask_path,force_mask*255)
            stress_map=cv2.cvtColor(stress_map, cv2.COLOR_BGR2RGB)

            cv2.imwrite(_stress_path, stress_map)
            ## display
            # plt.figure(1,figsize=(8,6))
            # plt.subplot(131)
            # plt.axis('off')
            # plt.imshow(sketch)
            # plt.subplot(132)
            # plt.axis('off')
            # plt.imshow(force_mask)
            # plt.subplot(133)
            # plt.imshow(stress_map)
            # plt.axis('off')
            # plt.show()



if __name__ == "__main__":
    BASE_DIR = '/home/wanchao/Projects/code/Projects_deng/sk2fabrication/colored_mesh_rendering'  #'/home/yudeng/data/projects'#os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    normalized_scale_com=256
    dataPath= BASE_DIR+'/data/sketch_out'
    d = DatasetSkStress(_dataPath=dataPath,_cate='airplane',_normalized_scale=normalized_scale_com,_phase='train')





