#shapeChairnew shapeTable shape Airplane
python train.py --batch_size 16 --gpu_ids 0 --model_name 'branch_normal_mask_D_normal_weighted' --lr_policy 'step' --n_epochs 50 --n_epochs_decay 50 --lr_decay_iters 1 --gan_mode 'l1' --netD 'multi_scale' --num_D 3 --netG 'branch_normal' --output_nc 4 --save_epoch_freq 1 --save_image_freq 1 --D_normal --cate_name 'cosegGuitar' --use_aug #--max_data_size 1

