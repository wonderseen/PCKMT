MODEL_PATH=pretrain_model/wmt19.de-en.ffn8192.pt
 
DSTORE_SIZE=3613350
DATA_PATH=data-bin/it
DATASTORE_PATH=save_datastore/it
PROJECT_PATH=.

mkdir -p $DATASTORE_PATH

CUDA_VISIBLE_DEVICES=6 python $PROJECT_PATH/save_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 1024 --dstore-fp16 \
    --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH
 
# 4096 and 1024 depend on your device and model separately

