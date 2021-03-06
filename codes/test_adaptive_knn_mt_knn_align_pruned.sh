 

DOMAIN=it # it medical koran law subtitles
DATASTORE_DOMAIN=${DOMAIN}
k=4
loss=1.84
GPU=0
BATCH_SIZE=64
PRUNE_METHOD=/prune_similar_ppl-0.9

declare -A DSTORE_SIZES_dict
DSTORE_SIZES_dict=([subtitles]="" [medical]="" [koran]=""  [it]="" [law]="" [merge]="" )
MODEL_PATH_SUFFIX=model_record_path/...
DATASTORE_PATH=save_datastore/...
DSTORE_SIZE=${DSTORE_SIZES_dict[${DATASTORE_DOMAIN}]}
MODEL_PATH=${MODEL_PATH_SUFFIX}/checkpoint.best_loss_${loss}.pt
OUTPUT_PATH=${MODEL_PATH_SUFFIX}/results 
PROJECT_PATH=.
DATA_PATH=data-bin/${DOMAIN}

mkdir -p "$OUTPUT_PATH" 

CUDA_VISIBLE_DEVICES=$GPU python $PROJECT_PATH/experimental_knn_align.py $DATA_PATH \
    --gen-subset test \
    --path "$MODEL_PATH" --arch transformer_knn_de_en_transfered_by_distance \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 \
    --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --batch-size $BATCH_SIZE \
    --tokenizer moses --remove-bpe --not-train-knn-compact-projection \
    --model-overrides "{
        'batch_size': $BATCH_SIZE,
        'load_knn_datastore': True, 
        'use_knn_datastore': True,
        'dstore_filename': '$DATASTORE_PATH',
        'dstore_size': $DSTORE_SIZE, 
        'dstore_fp16': True,
        'probe':32,
        'knn_sim_func': 'do_not_recomp_l2', 
        'use_gpu_to_search': True, 
        'move_dstore_to_mem': True, 'no_load_keys': True,
        'knn_temperature_type': 'fix', 
        'knn_temperature_value': 10,
        'k_lambda_net_dropout_rate': 0.0,
        'only_train_knn_parameter': False
    }" \
    | tee "$OUTPUT_PATH"/generate.txt 

grep ^S "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/src
grep ^T "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/ref
grep ^H "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp
grep ^D "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp.detok
