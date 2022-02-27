PROJECT_PATH=.
DOMAINS=(subtitles)
DSTORE_SIZES=(137803715)

gpu_ids=(0)

for idx in ${!gpu_ids[*]}
do
  LOG=
  DSTORE_PATH=
  CUDA_VISIBLE_DEVICES=${gpu_ids[$idx]} python ${PROJECT_PATH}/train_datastore_gpu.py \
    --dstore_mmap ${DSTORE_PATH} \
    --dstore_size ${DSTORE_SIZES[$idx]} \
    --faiss_index ${DSTORE_PATH}/knn_index \
    --ncentroids 4096 \
    --probe 32 \
    --dimension 64 \
    --dstore_fp16 \
    > ${LOG} 2>&1 &
    echo ${LOG}
    # --use-gpu \
done