DEVICE=0
DATA_SET='msra'
MODEL_CLASS='lebert-crf'
LR=1e-5
CRF_LR=1e-2
ADAPTER_LR=1e-3
PRETRAIN_MODEL='bert-base-chinese'
export CUDA_VISIBLE_DEVICES=${DEVICE}

python train.py \
    --device gpu \
    --output_path output \
    --add_layer 1 \
    --loss_type ce \
    --lr ${LR} \
    --crf_lr ${CRF_LR} \
    --myattention_lr 2e-2 \
    --lstm_lr 1e-3 \
    --adapter_lr ${ADAPTER_LR} \
    --weight_decay 0.01 \
    --eps 1.0e-08 \
    --epochs 10 \
    --batch_size_train 32 \
    --batch_size_eval 256 \
    --num_workers 0 \
    --eval_step 1000 \
    --max_seq_len 128 \
    --max_word_num  3 \
    --max_scan_num 3000000 \
    --data_path datasets/${DATA_SET}/ \
    --dataset_name ${DATA_SET} \
    --model_class ${MODEL_CLASS} \
    --pretrain_model_path ${PRETRAIN_MODEL} \
    --pretrain_embed_path /root/autodl-fs/imcs21/task/NER/LEBERT-NER/Downloads/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt \
    --seed 6 \
    --markup bio \
    --grad_acc_step 1 \
    --max_grad_norm 1.0 \
    --num_workers 0 \
    --warmup_proportion 0.1 \
    --load_word_embed \
    --do_eval \
    --do_train
