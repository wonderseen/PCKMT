## Abstract

Source codes of [ACL 2022-Efficient Cluster-Based k-Nearest-Neighbor Machine Translation](https://arxiv.org/abs/2204.06175).

The implement of our proposed PCKMT is build upon the research of:

- adaptive kNN-MT (Xin Zheng et. al. 2021) [[code]](https://github.com/zhengxxn/adaptive-knn-mt)
- Fairseq and Faiss developed by Facebook Research

## Requirement

For our case, the CUDA version is 10.1. We didn't check other versions yet.

- python >= 3.6
- faiss-gpu == 1.6.5
- torch == 1.5.0
- torch-scatter == 2.0.5

With these requirements, it is suggested to use the command to install this editable version (fairseq == 0.10.1):

```shell
pip install --editable ./
```

## Checkpoints

Our trained checkpoints, datastores and logs are provided: [baidu](https://pan.baidu.com/s/1CalRc6qcGlKQ86cprqqkEQ)
(Password: ckmt)


## Implement

Please follow the steps to reproduce experiments:

1. Follow the codebase of (Xin Zheng et. al. 2021) and download the [checkpoint of base De-En NMT model](https://github.com/pytorch/fairseq/blob/main/examples/wmt19/README.md) released by Facebook WMT 2019.
2. Similarly, download the [corpora and test sets](https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view) as illustrated by [Xin Zheng et. al. 2021](https://github.com/zhengxxn/adaptive-knn-mt).
3. Create the original datastore of adaptive kNN-MT.

```shell
. create_datastore.sh
```

4.  (Optional) Modify the script **prune_datastore.py** to fit your datastore (e.g., datadir, datastore size, etc. in the main() function) and then prune the datastore:

```shell
python prune_datastore.py
```

5. Train the Compact Network:

```shell
. knn_align.sh
```

6. Reconstruct the compressed datastore of CKMT

```shell
. create_datastore_knn_align.sh
```

7. Train the quantized index

```
. build_faiss_index_knn_align.sh
```

8. Train the CKMT model

```shell
. train_faiss_knn_align.sh
```

9. Evaluation

```shell
. test_adaptive_knn_mt_knn_align.sh
```

## Reference
If you use the source codes included here in your work, please cite the following paper:  
```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.06175,
  doi = {10.48550/ARXIV.2204.06175},
  url = {https://arxiv.org/abs/2204.06175},
  author = {Wang, Dexin and Fan, Kai and Chen, Boxing and Xiong, Deyi},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Efficient Cluster-Based k-Nearest-Neighbor Machine Translation},
  publisher = {arXiv},  
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```