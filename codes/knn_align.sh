MODEL_PATH=pretrain_model/wmt19.de-en.ffn8192.pt

COMPACT_DIM=64

gpu_ids=(0 0 0 1 1)
DSTORE_SIZES=(3613350 524400 6903320 19070000 153604142)
DOMAINS=(it koran medical law subtitles)


# It takes external time to duplicate the parser-dependent dataloader of fairseq
# into a global triple sampling dataloader. Instead, we make another parser-independent
# dataloader at function knn_training_iterator/TripletDatastoreSamplingDataset for
# training Compact Network, please see `fairseq/models/transformer_knn_projection.py`

# `batch_size_grid` and `valid_batch_size_grid` are both original parameters in
# adaptive knn-mt for bilingual corpus dataloading, while having no effect for
# knn_align.sh becuase the triple sampling dataloader is parser-independent.
# The only role of batch_size_grid for this script is that the training epoch is
# (roughly) computed as: bath_size * step // bilingual_corpus_size
batch_size_grid=(2 2 2 2 2)
valid_batch_size_grid=(64 64 64 64 64)
update_freq_grid=(1 1 1 1 1)
declare -A MAX_EPOCHS_dict
MAX_EPOCHS_dict=([koran]="30" [it]="70" [medical]="120" [law]="200" [subtitles]="200" )


for idx in ${!gpu_ids[*]}
do
  DOMAIN=${DOMAINS[$idx]}
  MAX_EPOCH=${MAX_EPOCHS_dict[$DOMAIN]}
  postfix=_nce_${COMPACT_DIM}

  # model storage
  MODEL_RECORD_PATH=model_record_path/${DOMAIN}/knn_transfered${postfix}

  # log
  TRAINING_RECORD_PATH=model_record_tensorboard_path/${DOMAIN}/knn_transfered${postfix}
  NOHUP_FILE=nohup-${DOMAIN}/knn_align${postfix}.txt

  # corpus
  DATA_PATH=data-bin/${DOMAIN}

  # datastore
  DATASTORE_PATH=save_datastore/${DOMAIN}
  DSTORE_SIZE=${DSTORE_SIZES[$idx]}


  rm $NOHUP_FILE
  rm -rf "$MODEL_RECORD_PATH"
  rm -rf "$TRAINING_RECORD_PATH"
  mkdir -p "$MODEL_RECORD_PATH"
  mkdir -p "$TRAINING_RECORD_PATH"


  # start
  PROJECT_PATH=.
  CUDA_VISIBLE_DEVICES=${gpu_ids[$idx]} nohup python \
  $PROJECT_PATH/fairseq_cli/knn_align.py \
    $DATA_PATH \
    --log-interval 100 --log-format simple \
    --arch transformer_knn_de_en_transfered_by_distance \
    --tensorboard-logdir "$TRAINING_RECORD_PATH" \
    --save-dir "$MODEL_RECORD_PATH" --restore-file "$MODEL_PATH" \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
    --validate-interval-updates 5000 --save-interval-updates 5000 \
    --keep-interval-updates 1 --max-update 200000 \
    --validate-after-updates 10000 --source-lang de --target-lang en \
    --save-interval 10000 --validate-interval 5000 \
    --keep-best-checkpoints 1 --no-epoch-checkpoints \
    --no-last-checkpoints --no-save-optimizer-state \
    --train-subset valid --valid-subset valid \
    --criterion triplet_ranking --label-smoothing 0.001 \
    --max-source-positions 1024 --max-target-positions 1024 \
    --batch-size "${batch_size_grid[$idx]}" --update-freq "${update_freq_grid[$idx]}" \
    --batch-size-valid "${valid_batch_size_grid[$idx]}" \
    --task translation --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --min-lr 3e-05 --lr 0.0003 --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau --lr-patience 5 --lr-shrink 0.5 \
    --patience 10 --max-epoch $MAX_EPOCH \
    --load-knn-datastore --dstore-filename $DATASTORE_PATH --use-knn-datastore \
    --dstore-fp16 --dstore-size $DSTORE_SIZE --probe 32 \
    --knn-sim-func do_not_recomp_l2 \
    --use-gpu-to-search --move-dstore-to-mem \
    --knn-lambda-type trainable --knn-temperature-type fix \
    --knn-temperature-value 10 --only-train-knn-parameter \
    --knn-k-type trainable --k-lambda-net-hid-size 32 --k-lambda-net-dropout-rate 0.0 \
    --max-k 4 --k 4 --label-count-as-feature --report-accuracy \
    --decoder-knn-compact-dim ${COMPACT_DIM} \
    > $NOHUP_FILE 2>&1 &
    echo $NOHUP_FILE
done
