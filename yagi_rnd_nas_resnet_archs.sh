#!/usr/bin/env sh
#$-pe gpu 1
#$-l gpu=1
#$-j y
#$-cwd
#$-V

# CUDA_VISIBLE_DEVICES="${gpu_num}" pythoneight_share.py \
python3 exp_main.py \
    --archs_per_task "${archs_per_task}" \
    --save_dir save_dir/resnet-"${time}"-"${num_train}"-"${archs_per_task}"-id"${id}"/ \
    --num_train "${num_train}" --archs_per_task "${archs_per_task}" \
    --arch_type resnet \
    >> "./log/resnet-${time}-${num_train}-${archs_per_task}-id${id}.log" 2>&1


# CUDA_VISIBLE_DEVICES="${gpu_num}" python3 random_weight_share.py \
#     --save_dir save_dir/"${num_train}"/ --batch_size "${batch_size}" \
#     --num_nodes "${num_nodes}" --num_layers "${num_layers}" \
#     --init_channels "${init_channels}" \
#     >> "./log/${time}_samples${num_train}_num_nodes${num_nodes}-id${id}.log" 2>&1
    