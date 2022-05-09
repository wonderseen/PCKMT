from typing import Sequence
import torch
import faiss
import numpy as np
from torch import functional
from torch_scatter import scatter
import time
import math
import faiss.contrib.torch_utils


class KNN_Dstore(object):

    def __init__(self, args, trg_vocab_size):
        self.half = args.fp16
        self.dimension = getattr(args, "dimension", args.decoder_output_dim)
        
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.temperature = args.knn_temperature_value
        self.use_gpu_to_search = args.use_gpu_to_search
        self.vocab_size = trg_vocab_size
        # self.only_use_max_idx = args.only_use_max_idx

        self.index = self.setup_faiss(args)
        self.time_for_retrieve = 0.
        self.retrieve_count = 0.
        self.time_for_setup_prob = 0.
        self.fcg = args.fcg

        # set lambda
        self.set_lambda(args)

        # set temperature
        self.temperature_type = args.knn_temperature_type
        if self.temperature_type == 'fix':
            self.temperature = args.knn_temperature_value # 10
        elif self.temperature_type == 'trainable':
            self.temperature = None
        else:
            self.temperature = None

        self.k_type = args.knn_k_type
        if self.k_type == 'fix':
            self.k = args.k

        elif self.k_type == 'trainable':

            assert args.max_k is not None

            self.max_k = args.max_k
            self.k = args.max_k
            # we first need to generate the mask for different k

            self.mask_for_distance = self.generate_neighbor_mask(args.max_k if args.max_k is not None else args.k)
            self.reduce_k = self.mask_for_distance.size(0)


        self.mask_for_label_count = self.generate_label_count_mask(args.max_k if args.max_k is not None else args.k)
        
        self.zero_knn_tgt_prob = torch.zeros(args.batch_size * 4, 1, args.max_k, self.vocab_size).cuda()
        

    def generate_neighbor_mask(self, max_k):

        k_mask = torch.empty((max_k, max_k)).fill_(9999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1

        # we only select 2's power here
        # [1 - 1, 2 - 1, 4 - 1, 8 - 1, ...]
        power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(self.max_k, 2)) + 1)])
        k_mask = k_mask[power_index]

        k_mask.requires_grad = False
        if torch.cuda.is_available():
            k_mask = k_mask.cuda()
        return k_mask

    def generate_label_count_mask(self, max_k):
        mask_for_label_count = torch.empty((max_k, max_k)).fill_(1)
        mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()

        if torch.cuda.is_available():
            mask_for_label_count = mask_for_label_count.cuda()

        mask_for_label_count.requires_grad = False

        return mask_for_label_count

    def get_label_count_segment(self,
                                tgt_idx: torch.Tensor,
                                relative=False):  # [B, S, K]
        """
        This function return the label counts for different range of k nearest neighbor
        [[0:0], [0:1], [0:2], ..., [0:K-1]]

        """

        B, S, K = tgt_idx.size()

        expand_tgt_idx = tgt_idx.unsqueeze(-2).expand(B, S, K, K)
        expand_tgt_idx = expand_tgt_idx.masked_fill(self.mask_for_label_count, value=-1)

        labels_sorted, _ = expand_tgt_idx.sort(dim=-1)  # [B, S, K, K]
        labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)  # [B, S, K]
        retrieve_label_counts[:, :, :-1] -= 1

        # if we want relative label count, i.e [1, 2, 3, 3, 4] -> [1, 1, 1, 0, 1]
        if relative:
            retrieve_label_counts[:, :, 1:] = retrieve_label_counts[:, :, 1:] - retrieve_label_counts[:, :, :-1]
            
        return retrieve_label_counts

    def get_label_count(self, tgt_idx: torch.Tensor):
        """
        This only return total label count for all neighbors
        """
        tgt_sorted, _ = tgt_idx.sort(dim=-1)
        tgt_sorted[:, :, 1:] *= ((tgt_sorted[:, :, 1:] - tgt_sorted[:, :, :-1]) != 0).long()
        retrieve_label_counts = tgt_sorted.ne(0).sum(-1).unsqueeze(-1)  # [B, S, 1]

        return retrieve_label_counts

    def set_lambda(self, args):

        if not hasattr(args, 'knn_lambda_type'):
            return

        self.lambda_type = args.knn_lambda_type

        if self.lambda_type == 'fix':
            self.lambda_value = args.knn_lambda_value

        if self.lambda_type == 'trainable':
            self.lambda_value = None  # not generate lambda value in this class

    def get_lambda(self, step=None, distance=None):

        if self.lambda_type == 'fix':
            return self.lambda_value

        elif self.lambda_type == 'trainable':
            return None

    def get_temperature(self):

        if self.temperature_type == 'fix':
            return self.temperature
        else:
            return None

    def setup_faiss(self, args):

        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()

        # index = faiss.read_index(args.dstore_filename + '/knn_index_pca', faiss.IO_FLAG_ONDISK_SAME_DIR)
        index = faiss.read_index(args.dstore_filename + '/knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)
        if self.use_gpu_to_search:
            print('put index from cpu to gpu')
            if torch.cuda.device_count() == 1: # the original implementation only used 1 GPU
                res = faiss.StandardGpuResources()
                self.res = res
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                index = faiss.index_cpu_to_gpu(res, 0, index, co)
            else: # multi-gpu version
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.usePrecomputed = False
                co.shard = True
                index = faiss.index_cpu_to_all_gpus(index, co, ngpu=torch.cuda.device_count())

        print('Reading datastore took {} s'.format(time.time() - start))
        print('the datastore is {}, size is {}, and dim is {} '.
              format(args.dstore_filename, self.dstore_size, self.dimension))

        index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int32')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '/keys.npy', dtype=np.float16, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '/vals.npy', dtype=np.int, mode='r',
                                  shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int32')
            print(args.dstore_filename)
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '/keys.npy', dtype=np.float32, mode='r',
                                      shape=(self.dstore_size, self.dimension))

            self.vals = np.memmap(args.dstore_filename + '/vals.npy', dtype=np.int, mode='r',
                                  shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # NOTE: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename + '/keys.npy',
                                                  dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                                  shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension),
                                     dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename + '/vals.npy',
                                              dtype=np.int, mode='r',
                                              shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int)

            if True:#self.use_gpu_to_search:
                self.vals = torch.from_numpy(self.vals)
                if torch.cuda.is_available():
                    print('put vals to gpu')
                    self.vals = self.vals.cuda()

            print('Loading to memory took {} s'.format(time.time() - start))

        return index

    def dist_func(self, d, k, q, function=None):

        if not function:
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            qsize = q.shape
            if self.metric_type == 'l2':
                knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                if self.half:
                    knns_vecs = knns_vecs.half()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                return -1 * l2
            return d

        if function == 'do_not_recomp_l2':
            return -1 * d

        raise ValueError("Invalid knn similarity function!")

    def get_knns(self, queries, k=None):
        # move query to numpy, if faiss version < 1.6.5
        if not self.use_gpu_to_search:
            current_deivce = queries.device
            queries = queries.detach().cpu().float().numpy()
        
        if k is None:
            dists, knns = self.index.search(queries, self.k)
        else:
            dists, knns = self.index.search(queries, k)

        if not self.use_gpu_to_search:
            dists = torch.tensor(dists).to(current_deivce)
            knns = torch.tensor(knns).to(current_deivce)
        return dists, knns

    def get_only_max_index(self, prob):
        # if we do not need a distribution, only use the max prob result
        prob = prob.zero_().scatter_(dim=-1, index=max_idx.unsqueeze(-1), value=1)

    def retrieve(self, queries):
        # queries  are [Batch, seq len, Hid Size]
        # retrieve
        B = queries.size(0)
        S = queries.size(1)
        H = queries.size(-1)
        device = queries.device

        dists, knns = self.get_knns(queries.contiguous().view(-1, H), k=self.k)  # [B * seq S, increase K]
        
        # move retireval results to torch tensor from numpy, if faiss version < 1.6.5
        # knns = torch.from_numpy(knns).to(device)
        # dists = torch.from_numpy(dists).to(device)  # [B * S, k]
        tgt_idx = self.vals[knns.to(device)].squeeze(-1).view(B, S, -1)  # [Batch size * Seq len, K]
        tgt_idx = tgt_idx  # [B, S, K]

        dists = dists.view(B, S, -1)  # [Batch, Seq len, k]
        knns = knns.view(B, S, -1)
        return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx}

    def calculate_select_knn_prob(self,
            knn_index: torch.Tensor,   # [B, S, K]
            tgt_index: torch.Tensor,   # [B, S, K]
            distance: torch.Tensor,    # [B, S, K]
            queries: torch.Tensor,     # [B, S, H]
            temperature: torch.Tensor, # [B, S, 1]
            knn_select_prob: torch.Tensor = None, # [B, S, Reduce K]
            is_test=False
        ):
        B, S, K = distance.size()

        R_K = knn_select_prob.size(-1)
        device = queries.device

        assert R_K == self.reduce_k or R_K == self.reduce_k + 1, (R_K,)

        re_compute_dists = self.dist_func(distance, knn_index, queries, function=self.sim_func) # [B, S, K]
        re_compute_dists = re_compute_dists.unsqueeze(-2).expand(B, S, self.reduce_k, K)

        # start = time.time()
        mask_for_distance_copy = self.mask_for_distance

        if self.sim_func in ['norm_angle']:
            re_compute_dists = re_compute_dists - mask_for_distance_copy + 1. # [B, S, self.reduce_k, K]
        else:
            re_compute_dists = re_compute_dists * mask_for_distance_copy  # [B, S, self.reduce_k, K]

        scaled_dists = re_compute_dists / temperature

        # if is_test:
        # TODO we may move matrix product by knn_mask to here to reduce memory usage, we already do this
        knn_weight = torch.softmax(scaled_dists, dim=-1)  # [B, S, self.reduce_k, K]
        weight_sum_knn_weight = torch.matmul(knn_select_prob.unsqueeze(-2), knn_weight).squeeze(-2).unsqueeze(-1)  # [B, S, K, 1]

        # NOTE: the original adaptive knn-mt implementation is slow as the large knn_tgt_prob tensor
        # is initialized repeatedly, hence we duplicated it as this:  
        if S != 1: # train
            knn_tgt_prob = torch.zeros(B, S, K, self.vocab_size).to(device)  # [B, S, K, Vocab Size]
        else: # test or val
            knn_tgt_prob = 0. * self.zero_knn_tgt_prob[:B, :S, :K, :self.vocab_size]

        tgt_index = tgt_index.unsqueeze_(-1)  # [B, S, K, 1]

        # start = time.time()
        scatter(src=weight_sum_knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)
        # print('scatter all prob, took {} s'.format(time.time() - start))
        
        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]
        
        return {'prob': prob}

        ## the original implementation
        # knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, R_K, K, 1]
        # knn_tgt_prob = torch.zeros(B, S, R_K, K, self.vocab_size).to(device)  # [B, S, R_K, K, Vocab Size]
        # tgt_index = tgt_index.unsqueeze_(-2).expand(B, S, R_K, K).unsqueeze(-1)  # [B, S, R_K, K, 1]
        # scatter(src=knn_weight, out=knn_tgt_prob, index=tgt_index, dim=-1)  # [B, S, R_K, K, Vocab Size]
        # prob = knn_tgt_prob.sum(dim=-2)  # [B, S, R_K, Vocab Size]
        # prob = torch.matmul(knn_select_prob.unsqueeze(-2), prob)  # [B, S, 1, Vocab Size]
        # prob = prob.squeeze(-2)
        # return {'prob': prob}

    def calculate_knn_prob(self,
                           knn_index: torch.Tensor,  # [B, S, K]
                           tgt_index: torch.Tensor,  # [B, S, K]
                           distance: torch.Tensor,  # [B, S, K]
                           queries: torch.Tensor,  # [B, S, H]
                           temperature: torch.Tensor,  # [B, S, 1]
                           ):

        bsz = queries.size(0)
        seq_len = queries.size(1)
        device = queries.device

        # update the dist and compute each neighbor weight, neg distance
        re_compute_dists = self.dist_func(distance, knn_index, queries, function=self.sim_func)  # [B, S, K]
        scaled_dists = re_compute_dists / temperature
        knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

        # set the target index for each neighbor
        knn_tgt_prob = torch.zeros(bsz, seq_len, self.k, self.vocab_size).to(device)  # [B, S, K, Vocab Size]
        tgt_index = tgt_index.unsqueeze_(-1)  # [B, S, K, 1]

        # implemented with pytorch_scatter
        scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)
        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        return {'prob': prob}

    def update_get_knn_seq_prob(self, queries):

        knn_search_result = self.retrieve(queries)

        if self.temperature_type == 'fix':
            final_result = self.calculate_knn_prob(knn_index=knn_search_result['knn_index'],
                                                   tgt_index=knn_search_result['tgt_index'],
                                                   distance=knn_search_result['distance'],
                                                   queries=queries,
                                                   temperature=self.temperature)

            return {'distance': knn_search_result['distance'],
                    'knn_index': knn_search_result['knn_index'],
                    'prob': final_result['prob'],
                    }


if __name__ == "__main__":
    class ARGS:
        fp16 = False
        decoder_embed_dim = 1024
        k = 64
        dstore_size = 524400
        faiss_metric_type = 'do_not_recomp_l2'
        knn_sim_func = 'do_not_recomp_l2'
        dstore_fp16 = True
        knn_temperature_value = 1.0
        indexfile = ''
        dstore_filename = ''
        no_load_keys = False
        probe = 32
        move_dstore_to_mem = True
        use_gpu_to_search = True
        trg_vocab_size = 42024


    args = ARGS()
    knn_store = KNN_Dstore(args=args, trg_vocab_size=args.trg_vocab_size)

    query = torch.randn(32 * 4, 1024)
    print('query size is {}', query.size())
    # dist, knn_idx = knn_store.get_knns(query)
    # print(dist.shape)  # [10000, 64]
    # print(knn_idx.shape)  # [10000, 64] 

    prob = knn_store.get_knn_prob(query)
    # print(prob.max(dim=-1)[0]) 
    # print(prob.max(dim=-1)[1])

    print('average time for retrieve neighbors, {} s'.format(knn_store.time_for_retrieve / knn_store.retrieve_count))
    print('average time for set the target prob for each neighbor'
          ' (need do scatter operation for (batch size * beam size * k, vocab size) tensor), {} s'
          .format(knn_store.time_for_setup_prob / knn_store.retrieve_count))
