postfix=_nce
gpu=2
OUTDOMAIN=law
DOMAIN=${OUTDOMAIN}
COMPACT_DIM=64


# corpus
DATA_PATH=data-bin/${DOMAIN} 


# datastore
declare -A DSTORE_SIZES_dict
DSTORE_SIZES_dict=([it]="3613350" [medical]="6903320" [koran]="524400" [law]="19070000" [wiki]="47987250" [subtitles]="153604142")
DSTORE_SIZE=${DSTORE_SIZES_dict[$OUTDOMAIN]}
DATASTORE_PATH=save_datastore/${OUTDOMAIN}/knn_transfered${postfix}
# rm -rf $DATASTORE_PATH
mkdir -p $DATASTORE_PATH

# model
MODEL_PATH=model_record_path/knn_transfered${postfix}_${COMPACT_DIM}/checkpoint_best.pt


# log
NOHUP_DIR=nohup-${DOMAIN}
mkdir -p ${NOHUP_DIR}


# start
PROJECT_PATH=.
CUDA_VISIBLE_DEVICES=$gpu python $PROJECT_PATH/save_datastore_knn_align.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 1024 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH \
    --decoder-knn-compact-dim ${COMPACT_DIM} --save-knn-compate-feature --not-train-knn-compact-projection \
    --dstore-filename $DATASTORE_PATH --create-knn-compact-projection \
    > ${NOHUP_DIR}/create_datastore${postfix}.txt 2>&1 &
    echo ${NOHUP_DIR}/create_datastore${postfix}.txt
