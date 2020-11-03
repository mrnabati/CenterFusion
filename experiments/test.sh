export CUDA_VISIBLE_DEVICES=1
cd src

## Perform detection and evaluation
python test.py ddd \
    --exp_id centerfusion \
    --dataset nuscenes \
    --val_split mini_val \
    --run_dataset_eval \
    --num_workers 4 \
    --nuscenes_att \
    --velocity \
    --gpus 0 \
    --pointcloud \
    --radar_sweeps 3 \
    --max_pc_dist 60.0 \
    --pc_z_offset -0.0 \
    --load_model ../models/centerfusion_e60.pth \
    --flip_test \
    # --resume \