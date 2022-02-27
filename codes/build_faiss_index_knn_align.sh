
DOMAIN=(medical law koran it )
DSTORE_SIZE=(6903142 19061383 524375 3602863 )

gpu_ids=(1 2)
PROJECT_PATH=.
for idx in ${!gpu_ids[*]}
do
  DSTORE_PATH=
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






# postfix=_nce
# DOMAIN=(subtitles)
# DSTORE_SIZE=(153604142)

# gpu_ids=(7)
# PROJECT_PATH=.
# for idx in ${!gpu_ids[*]}
# do
#   echo nohup-${DOMAIN[$idx]}/build_faiss_index${postfix}.txt
#   DSTORE_PATH=save_datastore/${DOMAIN[$idx]}/knn_transfered${postfix}
#   CUDA_VISIBLE_DEVICES=${gpu_ids[$idx]} python ${PROJECT_PATH}/train_datastore_gpu.py \
#     --dstore_mmap ${DSTORE_PATH} \
#     --dstore_size ${DSTORE_SIZE[$idx]} \
#     --faiss_index ${DSTORE_PATH}/knn_index \
#     --ncentroids 4096 \
#     --probe 32 \
#     --dimension 64 \
#     --dstore_fp16 
#     # > nohup-${DOMAIN[$idx]}/build_faiss_index${postfix}.txt 2>&1 &
# done