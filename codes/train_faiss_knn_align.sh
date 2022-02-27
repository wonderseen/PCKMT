DOMAINS=(law medical it subtitles koran)    
DSTORE_SIZES=(19061383 6903142 3602863 153604142 524375)
gpu_ids=(0 1 2 3 4)

BASE_DATASTORE_PATH=save_datastore
PROJECT_PATH=.  

max_k_grid=(4 4 4 4 4)
batch_size_grid=(16 16 16 16 16) # depending on your gpu memory
update_freq_grid=(1 1 1 1 1)
valid_batch_size_grid=(32 32 32 32 32)

for idx in ${!gpu_ids[*]}
do
  MODEL_PATH=...
  DATASTORE_PATH=${BASE_DATASTORE_PATH}/${DOMAINS[$idx]}/...
  MODEL_RECORD_PATH=model_record_path/${DOMAINS[$idx]}/...
  TRAINING_RECORD_PATH=model_record_tensorboard_path/${DOMAINS[$idx]}/...
  DATA_PATH=data-bin/${DOMAINS[$idx]}
  rm -rf "$MODEL_RECORD_PATH"
  rm -rf "$TRAINING_RECORD_PATH"
  mkdir -p "$MODEL_RECORD_PATH"
  mkdir -p "$TRAINING_RECORD_PATH"

  CUDA_VISIBLE_DEVICES=${gpu_ids[$idx]} python \
  $PROJECT_PATH/fairseq_cli/train.py \
    $DATA_PATH \
    --use-gpu-to-search \
    --log-interval 100 --log-format simple \
    --arch transformer_knn_de_en_transfered_by_distance \
    --tensorboard-logdir "$TRAINING_RECORD_PATH" \
    --save-dir "$MODEL_RECORD_PATH" --restore-file "$MODEL_PATH" \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
    --validate-interval-updates 100 --save-interval-updates 100 --keep-interval-updates 1 \
    --max-update 50000 --validate-after-updates 1000 \
    --save-interval 10000 --validate-interval 100 \
    --keep-best-checkpoints 1 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --train-subset valid --valid-subset valid --source-lang de --target-lang en \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.001 \
    --max-source-positions 1024 --max-target-positions 1024 \
    --batch-size "${batch_size_grid[$idx]}" --update-freq "${update_freq_grid[$idx]}" \
    --batch-size-valid "${valid_batch_size_grid[$idx]}" \
    --task translation \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --min-lr 3e-05 --lr 0.0003 --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau --lr-patience 5 --lr-shrink 0.5 \
    --patience 300 \
    --max-epoch 5000 \
    --load-knn-datastore --dstore-filename $DATASTORE_PATH --use-knn-datastore \
    --dstore-size "${DSTORE_SIZES[$idx]}" --probe 32 \
    --knn-sim-func do_not_recomp_l2 --no-load-keys \
    --move-dstore-to-mem \
    --knn-lambda-type trainable --knn-temperature-type fix \
    --only-train-knn-parameter --knn-k-type trainable \
    --k-lambda-net-hid-size 32 --k-lambda-net-dropout-rate 0.0 \
    --max-k "${max_k_grid[$idx]}" --k "${max_k_grid[$idx]}" \
    --label-count-as-feature --not-train-knn-compact-projection \
    --dstore-fp16 \
    --knn-temperature-value 10
done