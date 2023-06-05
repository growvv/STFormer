export CUDA_VISIBLE_DEVICES=5
cd ..
python tools/non_dist_train.py \
    --dataname radar\
    -m SimVP \
    --model_type gsta \
    --lr 1e-3  \
    --ex_name radar_simvp_gsta \
    --data_root /home/lfr/mnt/Radar_Data \
    --batch_size 1 \
    --use_gpu 0 \
    --val_batch_size 1 \
    --pre_seq_length 10 \
    --aft_seq_length 10 \
    --total_length 20 \
    --res_dir   /home/lfr/mnt/Radar_Data/simvp/SimVP_gSTA/test_load45_137_46_2 \
    --config_file ./configs/radar/SimVP_gSTA.py \
    --epoch 1 \
    --log_step 1\
    --save_step 1 \
    --is_train 0 \
    --load 1 \
    --weight_path /home/lfr/mnt/Radar_Data/simvp/SimVP_gSTA/45/radar_simvp_gsta/checkpoints \
    --pretrain_epoch 137 \
    --train_data_paths small_2000_10.npz \
    --valid_data_paths small_2000_10.npz \
    --test_data_paths small_2000_10.npz \