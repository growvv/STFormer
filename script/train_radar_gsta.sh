export CUDA_VISIBLE_DEVICES=8
cd ..
python tools/non_dist_train.py \
    --dataname radar\
    -m SimVP \
    --model_type gsta \
    --lr 1e-3  \
    --ex_name radar_simvp_gsta \
    --data_root /home/lfr/mnt/Radar_Data \
    --batch_size 1 \
    --pre_seq_length 10 \
    --aft_seq_length 10 \
    --total_length 20 \
    --res_dir   /home/lfr/mnt/Radar_Data/simvp/SimVP_gSTA/323 \
    --config_file ./configs/radar/SimVP_gSTA.py \
    --epoch 100 \
    --log_step 1\
    --save_step 1 \
    --is_train 1 \
    --load 0 \
    --weight_path /home/lfr/mnt/Radar_Data/simvp/SimVP_gSTA/220_02/radar_simvp_gsta/checkpoints \
    --pretrain_epoch 98 \