### evaluation ---------------------------------------------------------
####cates: aniHead_clear cosegGuitar cosegVase psbFish psbFourleg shapeMug shapeSegRocket shapeSegSkateboard shapeTable shapeChair
python test.py --batch_size 80 --gpu_ids 0 --model_name 'branch_normal_mask_D_normal_weighted' --netG branch_normal --eval --epoch 60 --output_nc 4 --cate_name 'cosegGuitar' --max_data_size 1000000000000000000000000000

#
### Ablation Study -------------------------------------------
## full method
#python test_branch.py --batch_size 100 --gpu_ids 0 --model_name 'branch_normal_mask_D_normal_weighted' --netG branch_normal --eval --epoch 5 --output_nc 4 --cate_name 'shapeAirplane' --max_data_size 500
## no point mask
# python test_branch.py --batch_size 20 --gpu_ids 0 --model_name 'branch_normal_mask_D_normal' --netG branch_normal --eval --epoch 5 --output_nc 4 --cate_name 'shapeAirplane' #--max_data_size 500
## no shape mask
# python test_branch.py --batch_size 20 --gpu_ids 0 --model_name 'branch_normal_D_normal' --netG branch_normal --eval --epoch 5 --output_nc 3 --cate_name 'shapeAirplane' #--max_data_size 500
## no G
#python test_branch.py --batch_size 50 --gpu_ids 0 --model_name 'branch_normal_mask_D_normal_weighted_s80' --netG branch_normal --eval --epoch 100 --output_nc 4 --cate_name 'cosegGuitar' --max_data_size 1000000000
## coordinates_input
# python test_branch_coordinates_input.py --batch_size 50 --gpu_ids 0 --model_name 'branch_normal_mask_D_normal_coordinates_input' --netG 'branch_normal_coordinates_input' --eval --epoch 4 --input_nc 1 --output_nc 4 --cate_name 'shapeChair' --max_data_size 1000000000

### ------------------------------------------------------------