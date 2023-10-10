export PYTHONPATH=/paddle/PaddleNLP/
export XPU_PADDLE_FC_INT32_WITH_LL=1
#export XPU_PADDLE_L3_SIZE=40060288

#export XPU_LLAMA_FFN=True
export XPU_TRANSFORMER_ENGINE=True
export LD_LIBRARY_PATH=/paddle/baidu/xpu/fast_paddle/build/kernel_plugin/so:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH  #rdma path
#export XPU_DEBUG=0X20
#export XPUAPI_DEBUG=0x21
#export XPU_PADDLE_DEBUG=1
#export GLOG_v=10
#export BKCL_DEBUG=1

#export BKCL_PCIE_RING=1
#export BKCL_RING_BUFFER_SIZE=33554432
#export BKCL_RING_BUFFER_SIZE=8388608
#export BKCL_CCIX_BUFFER_GM=1
export FLAGS_fuse_parameter_memory_size=128
export FLAGS_fuse_parameter_groups_size=128

unset PADDLE_MASTER
unset PADDLE_NNODES
unset PADDLE_JOB_ID
unset MASTER_ADDR
unset MASTER_PORT
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK

export BKCL_CCIX_RING=1
export BKCL_CCIX_BUFFER_GM=1
export BKCL_SOCKET_IFNAME=xgbe0   # ifconfig看下带公网ip的网口名，常见的如bond0、xgbe0、eth0等
export BKCL_USE_RDMA=1                 # 多机RDMA开关
export BKCL_RDMA_FORCE_TREE=1          # 多机间通过tree的方式进行RDMA通信，2/4/8机场景建议开启
export BKCL_RDMA_NICS=xgbe2,xgbe3,xgbe4,xgbe5,xgbe6,xgbe7,xgbe8,xgbe9
export BKCL_TREE_THRESHOLD=1
export BKCL_FORCE_SYNC=1


#rm -rf autotune_result_fc_at*
unset XPU_FC_AUTOTUNE
export XPU_FC_AUTOTUNE_FILE="./autotune_result_fc.txt"
unset XPU_FC_AUTOTUNE_WRITEBACK

# down loaddata
#mkdir data
#cd data
#wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
#wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
#cd ..

# 根据需要修改如下的多机ip地址
    #--master  10.93.200.26:47789  --ips="10.93.200.75,10.93.200.26" --nnodes 2 \
task_name="llama_pretrain_13b_final"
python -u  -m paddle.distributed.launch \
        --master  10.93.200.75:47789  --ips="10.93.200.11,10.93.200.25,10.93.200.26,10.93.200.75" --nnodes 4 \
        --devices "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-13b" \
    --tokenizer_name_or_path "facebook/llama-13b" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --fuse_attention_qkv 1 \
    --fuse_attention_ffn 1 \
    --tensor_parallel_degree 8 \
    --pipeline_parallel_degree 4 \
    --virtual_pp_degree 1 \
    --sequence_parallel 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 1\
    --recompute 0 \
    --recompute_granularity core_attn \
    --do_train \
    --pipeline_parallel_config "disable_partial_send_recv" \
    --device "xpu"