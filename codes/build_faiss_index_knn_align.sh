
postfix=_nce
DOMAIN=(law it koran medical subtitles)
DSTORE_SIZE=(19061383 3602863 524375 6903142 153604142)
gpu_ids=(0 1 2 3 4)

PROJECT_PATH=.
for idx in ${!gpu_ids[*]}
do
  DSTORE_PATH=save_datastore/${DOMAIN[$idx]}/knn_transfered${postfix}
  CUDA_VISIBLE_DEVICES=${gpu_ids[$idx]} python ${PROJECT_PATH}/train_datastore_gpu.py \
    --dstore_mmap ${DSTORE_PATH} \
    --dstore_size ${DSTORE_SIZE[$idx]} \
    --faiss_index ${DSTORE_PATH}/knn_index \
    --ncentroids 4096 \
    --probe 32 \
    --dimension 64 \
    --dstore_fp16 \
    --use-gpu \
    > nohup-${DOMAIN[$idx]}/build_faiss_index${postfix}.txt 2>&1 &
  echo nohup-${DOMAIN[$idx]}/build_faiss_index${postfix}.txt
done
