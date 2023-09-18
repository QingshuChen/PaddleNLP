export PYTHONPATH=/paddle/PaddleNLP/:$PYTHONPATH
export XPU_PADDLE_FC_INT32_WITH_LL=1
export FLAGS_use_stride_kernel=0
#    --fp16  \
#    --fp16_opt_level "O2"  \

export BKCL_CCIX_RING=1
#export BKCL_RING_BUFFER_SIZE=33554432
#export BKCL_RING_BUFFER_SIZE=8388608
export BKCL_RING_BUFFER_SIZE=4194304
#export BKCL_CCIX_BUFFER_GM=1
export BKCL_RING_BUFFER_SIZE=524288
export FLAGS_fuse_parameter_memory_size=128
export FLAGS_fuse_parameter_groups_size=128
export BKCL_SOCKET_FORCE_TREE=1

unset PADDLE_MASTER
unset PADDLE_NNODES
unset PADDLE_JOB_ID

export BKCL_SOCKET_IFNAME=bond0   # ifconfig看下带公网ip的网口名，常见的如bond0、xgbe0、eth0等
export BKCL_USE_RDMA=1                 # 多机RDMA开关
export BKCL_RDMA_FORCE_TREE=1          # 多机间通过tree的方式进行RDMA通信，2/4/8机场景建议开启
export BKCL_RDMA_NICS=xgbe2,xgbe3,xgbe4,xgbe5,xgbe6,xgbe7,xgbe8,xgbe9
export BKCL_TREE_THRESHOLD=1

task_name_or_path="llama_hybid"
python -u  -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name_or_path""_log" \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-13b" \
    --tokenizer_name_or_path "facebook/llama-13b" \
    --input_dir "./data" \
    --output_dir "output/$task_name_or_path" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --lr_scheduler_type "cosine" \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --dataloader_num_workers 1 \
    --sharding "stage3" \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 1\
    --recompute 1 \
    --do_train \
    --do_eval \
    --device "xpu"
