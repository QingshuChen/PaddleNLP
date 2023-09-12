export PYTHONPATH=/paddle/PaddleNLP/
export XPU_PADDLE_FC_INT32_WITH_LL=1
#export XPU_PADDLE_L3_SIZE=40060288
export XPU_GPT3_FFN=True
export XPU_LLAMA_FFN=True
export LD_LIBRARY_PATH=/paddle/baidu/xpu/fast_paddle/build/kernel_plugin/so:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH  #rdma path
#export XPU_DEBUG=0X20
#export XPUAPI_DEBUG=0x21
#export XPU_PADDLE_DEBUG=1
#export GLOG_v=10

#export BKCL_PCIE_RING=1
export BKCL_CCIX_RING=1
export BKCL_SOCKET_IFNAME=xgbe0
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

export BKCL_SOCKET_IFNAME=xgbe0   # ifconfig看下带公网ip的网口名，常见的如bond0、xgbe0、eth0等
export BKCL_USE_RDMA=1                 # 多机RDMA开关
export BKCL_RDMA_FORCE_TREE=1          # 多机间通过tree的方式进行RDMA通信，2/4/8机场景建议开启
export BKCL_RDMA_NICS=xgbe2,xgbe3,xgbe4,xgbe5,xgbe6,xgbe7,xgbe8,xgbe9
export BKCL_TREE_THRESHOLD=1

# down loaddata
#mkdir data
#cd data
#wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
#wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
#cd ..

# 根据需要修改如下的多机ip地址
    #--master  10.93.200.26:47789  --ips="10.93.200.75,10.93.200.26" --nnodes 2 \
task_name="gpt_pretrain"
#python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
python -u  -m paddle.distributed.launch \
 --master  10.93.200.89:$1  --ips="10.93.200.89,10.93.200.90,10.93.200.76,10.93.200.204" --nnodes 4 \
    	--devices "0,1,2,3,4,5,6,7" \
run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path gpt3-13B-en \
    --tokenizer_name_or_path gpt3-13B-en \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 512 \
    --per_device_train_batch_size 4 \
    --tensor_parallel_degree 8 \
    --pipeline_parallel_degree 4 \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --recompute 0 \
    --gradient_accumulation_steps 64 \
    --do_train \
    --device "xpu"
