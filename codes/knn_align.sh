MODEL_PATH=pretrain_model/wmt19.de-en.ffn8192.pt

DSTORE_SIZES=(3613350)
DOMAINS=(it)
MAX_UPDATES=(200000)

compact_dim=64
max_k_grid=(2)
batch_size_grid=(2)
update_freq_grid=(1)
valid_batch_size_grid=(64)
gpu_ids=(0)

PROJECT_PATH=.

for idx in ${!gpu_ids[*]}
do
  DOMAIN=${DOMAINS[$idx]}
  NOHUP_FILE=
  rm $NOHUP_FILE
  
  DSTORE_SIZE=${DSTORE_SIZES[$idx]}
  MODEL_RECORD_PATH=model_record_path/...
  TRAINING_RECORD_PATH=model_record_tensorboard_path/...
  DATA_PATH=data-bin/${DOMAIN}
  DATASTORE_PATH=save_datastore/${DOMAIN}

  rm -rf "$MODEL_RECORD_PATH"
  rm -rf "$TRAINING_RECORD_PATH"
  mkdir -p "$MODEL_RECORD_PATH"
  mkdir -p "$TRAINING_RECORD_PATH"

  CUDA_VISIBLE_DEVICES=${gpu_ids[$idx]} python \
  $PROJECT_PATH/fairseq_cli/knn_align.py \
    $DATA_PATH \
    --log-interval 100 --log-format simple \
    --arch transformer_knn_de_en_transfered_by_distance \
    --tensorboard-logdir "$TRAINING_RECORD_PATH" \
    --save-dir "$MODEL_RECORD_PATH" --restore-file "$MODEL_PATH" \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
    --validate-interval-updates 5000 --save-interval-updates 5000 --keep-interval-updates 1 \
    --max-update ${MAX_UPDATES[$idx]} --validate-after-updates 10000 \
    --save-interval 10000 --validate-interval 5000 \
    --keep-best-checkpoints 1 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --train-subset valid --valid-subset valid --source-lang de --target-lang en \
    --criterion triplet_ranking --label-smoothing 0.001 \
    --max-source-positions 1024 --max-target-positions 1024 \
    --batch-size "${batch_size_grid[$idx]}" --update-freq "${update_freq_grid[$idx]}" \
    --batch-size-valid "${valid_batch_size_grid[$idx]}" \
    --task translation \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --min-lr 3e-05 --lr 0.0003 --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau --lr-patience 5 --lr-shrink 0.5 \
    --patience 60 --max-epoch 20000 \
    --load-knn-datastore --dstore-filename $DATASTORE_PATH --use-knn-datastore \
    --dstore-fp16 --dstore-size $DSTORE_SIZE --probe 32 \
    --knn-sim-func do_not_recomp_l2 \
    --use-gpu-to-search --move-dstore-to-mem \
    --knn-lambda-type trainable --knn-temperature-type fix --knn-temperature-value 1 --only-train-knn-parameter \
    --knn-k-type trainable --k-lambda-net-hid-size 32 --k-lambda-net-dropout-rate 0.0 \
    --max-k "${max_k_grid[$idx]}" --k "${max_k_grid[$idx]}" \
    --label-count-as-feature --report-accuracy \
    --decoder-knn-compact-dim ${compact_dim} \
    > $NOHUP_FILE 2>&1 &
    echo $NOHUP_FILE
done
