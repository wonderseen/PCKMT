'''
the script to prune the datastore
'''
import logging
import random  
from typing import List, Dict
import warnings
from tqdm import tqdm
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from sklearn.cluster import Birch, DBSCAN, SpectralClustering
from multiprocessing import Pool
from collections import Counter
import os
import math
import shutil

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=sklearn.exceptions.ConvergenceWarning)
logging.basicConfig(level = logging.INFO,format = '%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




# cluster all key clusters w.r.t. each vocab
use_cluster = True # default is true
use_valset_to_retrieve = False
if use_valset_to_retrieve:
    '''
    NOTE: Duplicated for CKMT.
    This is semi-supervised pruning for future work. 
    When all_vocab_considered = True, a new pruned datastore should consider all vocabs 
    and all clusters of the general datastore. when all_vocab_considered = False, we mean that
    we only make selection on seen clusters of valid data when build the pruned datastore.
    '''
    gmm_pruned_on_seen_vocabs = True
    gmm_pruned_on_unseen_vocabs = True
    all_vocab_considered = gmm_pruned_on_seen_vocabs and gmm_pruned_on_unseen_vocabs
    valset_similarity_threshold = -200.
else:
    gmm_pruned_on_seen_vocabs = False
    gmm_pruned_on_unseen_vocabs = False
    all_vocab_considered = False
    valset_similarity_threshold = -1000000.


def precision_score(label, prediction):
    tp = label & prediction.astype(np.int)     
    precision = tp.sum() / prediction.sum()
    return precision


def recall_score(label, prediction):
    tp = label & prediction.astype(np.int)     
    recall = tp.sum() / label.sum()
    return recall


def calc_medoid(X, Y, f=2):
    n = len(X)
    m = len(Y)
    dist_mat = np.zeros((m, n))
    # compute distance matrix
    for j in range(n):
        center = X[j, :]
        for i in range(m):
            if i != j:
                dist_mat[i, j] = np.linalg.norm(Y[i, :] - center, ord=f)

    medoid_id = np.argmin(dist_mat.sum(axis=0))  # sum over y
    return medoid_id, X[medoid_id, :]


def draw_vocab_distribution(dictionary, distribution, filename_prefix: str = ''):
    dictionary = list(
        map(lambda x:x[0], sorted(list(zip(dictionary, distribution)),
            key=lambda d: d[1], reverse=True)))
    distribution.sort(reverse=True)
    dictionary = dictionary[:40]
    distribution = distribution[:40]

    x = range(len(dictionary))
    y = distribution
    plt.plot(x, y, marker='o', mec='r')
    # plt.legend()
    plt.xticks(x, dictionary, rotation=90)
    plt.xlabel("vocab")
    plt.ylabel("frequency")
    plt.title("Vocab Frequencies of %s Domain" % filename_prefix)
    plt.show()
    plt.savefig('vocab_freq_%s.png' % filename_prefix, dpi=200)
    # plt.close()


def get_mt_datastore(
    dstore_filename: str,
    dstore_fp16: bool,
    dstore_size: int,
    fea_size: int,
    mode: str = 'r'):
    assert mode in ['r', 'w+']
    logger.info('%s %s from %s' % (
        'Saving' if mode == 'w+' else 'Reading',
        'fp16' if dstore_fp16 else 'fp32',
        dstore_filename))
    
    if dstore_fp16:
        dstore_keys = np.memmap(dstore_filename + '/keys.npy',
            dtype=np.float16,
            mode=mode,
            shape=(dstore_size, fea_size))
    else:
        dstore_keys = np.memmap(dstore_filename + '/keys.npy',
            dtype=np.float32,
            mode=mode,
            shape=(dstore_size, fea_size))
    dstore_tgt_ids = np.memmap(dstore_filename + '/vals.npy',
        dtype=np.int64,
        mode=mode,
        shape=(dstore_size, 1))
    dstore_tgt_lens = np.memmap(dstore_filename + '/tgt_lens.npy',
        dtype=np.int64,
        mode=mode,
        shape=(dstore_size, 1))
    dstore_src_lens = np.memmap(dstore_filename + '/src_lens.npy',
        dtype=np.int64,
        mode=mode,
        shape=(dstore_size, 1))
    dstore_tgt_id_4_gram = np.memmap(dstore_filename + '/vals_4_gram.npy',
        dtype=np.int64,
        mode=mode,
        shape=(dstore_size, 4))
    dstore_tgt_id_4_gram_prob = np.memmap(dstore_filename + '/vals_4_gram_probs.npy',
        dtype=np.float32,
        mode=mode,
        shape=(dstore_size, 4))                     
    dstore_tgt_entropy = np.memmap(dstore_filename + '/vals_entropy.npy',
        dtype=np.float32,
        mode=mode,
        shape=(dstore_size, 1))
    return dstore_keys, dstore_tgt_ids, dstore_tgt_lens, dstore_src_lens, \
            dstore_tgt_id_4_gram, dstore_tgt_id_4_gram_prob, dstore_tgt_entropy


def random_sample(keys:List, nums: int = 1000000) -> List:
    assert type(keys) in [list, np.ndarray], type(keys)
    if isinstance(keys, List):
        if len(keys) > nums:
            return random.sample(keys, nums)
        else:
            return keys
    else:
        if keys.shape[0] > nums:
            return keys[np.random.choice(keys.shape[0], nums, replace=False)]
        else:
            return keys


def middle_k_idx(idxs: np.array, values: List[float], k:int = None) -> np.array:
    '''
    values: [0.2, 0.5, 0.323,  0.9,   0.1  ]
    idxs:   [10,  49,    29,  1999, 3020302]
    we sort zip(idxs, values) in the sort of values, and get k middle sorted-idxs

    sorted:
    values: [  0.1,   0.2, 0.323,  0.5,  0.9]
    idxs:   [3020302,  10,   29,   49,  1999]

    if k == 1, return [29]
    if k == 2, return [10, 29]
    if k == 3, return [10, 29, 49]
    etc.
    '''
    n = len(values)
    if n <= k:
        return idxs
    
    idxs = np.array(idxs)
    values = np.array(values)

    assert values.shape[0] == idxs.shape[0]

    top = (n - k) // 2 + k
    top_ind = np.argpartition(values, top)[:top]
    top_values = values[top_ind]
    top_idxs = idxs[top_ind]

    middle_k_ind = np.argpartition(top_values, -k)[-k:]
    middle_k_idxs = top_idxs[middle_k_ind]
    return middle_k_idxs


def ppl_split_and_sample(
    ppl_group: np.array,
    sample_rate: float = 0.3,
    translation_cost_threshold : float = 1.5,
    minimum_sample: int = 2
    ):
    if ppl_group.shape[0] > 1e4:
        # linear cluster (faster, not optical but acceptable)
        sc = Birch(n_clusters=None, threshold=translation_cost_threshold)#, branching_factor=256)
        clustering = sc.fit(ppl_group[:, None]) # train
        labels = clustering.labels_

        ppl_clusters = [[] for _ in range(labels.max() + 1)]
        for n in range(labels.shape[0]):
            if labels[n] == -1: ## isolated node
                continue
            ppl_clusters[labels[n]].append(n)
        for i, clusters in enumerate(ppl_clusters):
            clusters = np.array(clusters)
            sample_nums = max(min(minimum_sample, clusters.shape[0]), int(sample_rate * clusters.shape[0]))
            clusters = random_sample(clusters, sample_nums)
            # clusters = middle_k_idx(clusters, ppl_group[clusters], k=sample_nums)
            ppl_clusters[i] = clusters
            
        for n in range(labels.shape[0]):
            if labels[n] == -1: ## isolated node
                ppl_clusters.append(np.array([n], dtype=np.int))
        ppl_clusters = [ppl_index for ppl_index in ppl_clusters if ppl_index.shape[0] > 0]
        mask = np.hstack(ppl_clusters)
        assert mask.shape[0] <= ppl_group.shape[0]
        return mask
    else:
        # affinity greedy searching
        ppl_affinity = ppl_group[None] - ppl_group[:, None]
        ppl_similar = np.abs(ppl_affinity) <= translation_cost_threshold
        ppl_idx_clusters = []

        idx_empty = np.arange(ppl_similar.shape[0])
        while ppl_similar.sum() != 0.:
            ppl_similar_numbers = ppl_similar.astype(np.float32).sum(-1)
            ppl_max_similar_idx = np.argmax(ppl_similar_numbers)
            select_mask = ppl_similar[ppl_max_similar_idx]
            ppl_idx_clusters.append(idx_empty[select_mask])
            ppl_similar = ppl_similar[~select_mask]
            ppl_similar = ppl_similar[:, ~select_mask]
            idx_empty = idx_empty[~select_mask]

        for i, clusters in enumerate(ppl_idx_clusters):
            sample_nums = max(min(minimum_sample, clusters.shape[0]), int(sample_rate * clusters.shape[0]))
            clusters = random_sample(clusters, sample_nums)
            # clusters = middle_k_idx(clusters, ppl_group[clusters], k=sample_nums)
            ppl_idx_clusters[i] = clusters

        mask = np.hstack(ppl_idx_clusters)
        assert mask.shape[0] <= ppl_group.shape[0], (ppl_idx_clusters)
        return mask


def n_gram_prune_thread_inner_table_n_gram_idx_dict(
    table_n_gram_idx_dict: Dict,
    prune_style: str,
    mininum_sample: int,
    sample_rate: float,
    n_gram_uniform_ppl = None,
    tgt_entropy = None,
    ):
    for n_gram_str_symbol, np_idxs in tqdm(table_n_gram_idx_dict.items()):
    # for n_gram_str_symbol in tqdm(table_n_gram_idx_dict_keys):
        # np_idxs = table_n_gram_idx_dict[n_gram_str_symbol]

        selected_num = max(mininum_sample, int(sample_rate * np_idxs.shape[0]))

        # --- too sparse, we do not prune it
        if np_idxs.shape[0] <= selected_num:
            continue

        # --- 1. random selection
        if prune_style == 'random':
            table_n_gram_idx_dict[n_gram_str_symbol] = random_sample(np_idxs, selected_num)

        # --- 2. ppl pruning
        elif 'ppl' in prune_style:
            ppl_group = n_gram_uniform_ppl[np_idxs]

            if prune_style == 'prune_high_ppl':
                # --- get lower ppl
                mask = np.argpartition(ppl_group, selected_num)[:selected_num] 
            elif prune_style == 'prune_low_ppl':
                # --- get higher ppl
                mask = np.argpartition(ppl_group, -selected_num)[-selected_num:] 
            elif prune_style == 'prune_half_low_half_high_ppl':
                # --- get half higher and half lower ppl
                mask1 = np.argpartition(ppl_group, selected_num // 2)[:selected_num // 2] # half lower ppl
                mask2 = np.argpartition(ppl_group, -selected_num // 2)[-selected_num // 2:] # half higher ppl
                mask  = np.concatenate((mask1, mask2), axis=0)
            elif prune_style == 'prune_similar_ppl':
                # --- get similar-ppl pruned
                mask = ppl_split_and_sample(ppl_group, sample_rate=sample_rate)
            table_n_gram_idx_dict[n_gram_str_symbol] = np_idxs[mask]

        # --- 3. entropy pruning
        elif 'entropy' in prune_style:
            entropy_group = tgt_entropy[np_idxs]
            if prune_style == 'prune_high_entropy':
                # --- get lower entropy
                mask = np.argpartition(entropy_group, selected_num)[:selected_num] 
            elif prune_style == 'prune_low_entropy':
                # --- get higher entropy
                mask = np.argpartition(entropy_group, -selected_num)[-selected_num:] 
            elif prune_style == 'prune_half_low_half_high_entropy':
                # --- get half higher and half lower entropy
                mask1 = np.argpartition(entropy_group, selected_num // 2)[:selected_num // 2] # half lower entropy
                mask2 = np.argpartition(entropy_group, -selected_num // 2)[-selected_num // 2:] # half higher entropy
                mask  = np.concatenate((mask1, mask2), axis=0)
            elif prune_style == 'prune_similar_entropy':
                # --- get similar-entropy pruned
                mask = ppl_split_and_sample(entropy_group, sample_rate=sample_rate)
            table_n_gram_idx_dict[n_gram_str_symbol] = np_idxs[mask]

        # --- 4. TODO length count pruning
        else:
            raise NotImplementedError('not implemented prune_style = %s' % prune_style)

    return table_n_gram_idx_dict


def collect_pruned_n_grams_thread(table_n_gram_idx_dict: Dict):
    len_d = len(table_n_gram_idx_dict)
    val_list = [[] for _ in range(len_d)]
    dbidx_list = [[] for _ in range(len_d)]
    for i, (n_gram_str_symbol, np_idxs) in enumerate(table_n_gram_idx_dict.items()):
        np_idxs = table_n_gram_idx_dict[n_gram_str_symbol]

        # --- slow 
        # '30-23-40' -> [30, 23, 40], the first element is the final token vocab id of this phrase
        # vocab_id = int(n_gram_str_symbol.split('.')[0])

        # --- fast solution
        # '30.0557223434982' -> the integer part is the final token vocab id of this phrase
        vocab_id = int(n_gram_str_symbol)


        val_list[i] = [vocab_id] * np_idxs.shape[0]
        dbidx_list[i] = np_idxs.tolist()
        # tgt_lens_list[i] = general_tgt_lens[np_idxs].tolist()
        # src_lens_list[i] = general_src_lens[np_idxs].tolist()
    return val_list, dbidx_list, None, None #tgt_lens_list, src_lens_list



def n_gram_prune(
        general_dstore_size: int,
        general_keys: np.array, # [dstore_size, fea_dim]
        general_vocab_ids_4_gram: np.array, # [dstore_size, 4]
        general_tgt_lens, # [dstore_size, 1]
        general_src_lens, # [dstore_size, 1]
        general_tgt_4_gram_probs, # [dstore_size, 4]
        general_tgt_entropy, # [dstore_size, 1]
        dictionary: List[str], 
        n_of_n_gram: int = 2,
        sample_rate: float = 0.1,
        mininum_sample: int = 2,
        prune_style: str = 'random'
    ):
    start = time.time()
    logger.info('start n_gram_prune using %s method...' % prune_style)

    # --- compute best ppl for phrases ending with the same token
    '''e.g., for a phrase 'it is a lovely dog' (which ends with 'dog'),
    we collect normalized ppls of all n-grams:
      - ppl of 'dog' = ppls[:1] / 1
      - ppl of 'lovely dog' = ppls[:2].sum() / 2 
      - ppl of 'a lovely dog' = ppls[:3].sum() / 3
      - ppl of 'is a lovely dog' = ppls[:4].sum() / 4
      - ppl of 'it is a lovely dog' = ppls[:5].sum() / 5
    '''
    ppl_mask = (general_vocab_ids_4_gram != 0).astype(np.float32) # padding
    n_gram_uniform_ppl = -np.log(general_tgt_4_gram_probs * ppl_mask + 1e-5)
    n_gram_uniform_ppl = np.concatenate([n_gram_uniform_ppl[:, :i+1].sum(-1, keepdims=True) / (i+1) \
            for i in range(n_gram_uniform_ppl.shape[-1])], axis=-1)
    logger.info('n_gram_uniform_ppl established.')

    # --- get the translation entropy of each token
    tgt_entropy = general_tgt_entropy[:, 0]
    logger.info('tgt_entropy established.')
    
    # --- determine n for n-gram
    if 1 <= n_of_n_gram <= 4:
        n_gram_uniform_ppl = np.min(n_gram_uniform_ppl[:, :n_of_n_gram], axis=-1)
        # n_gram_uniform_ppl_min_arg = np.minarg(n_gram_uniform_ppl[:, :n_of_n_gram], axis=-1)

        '''fast solution
        n_gram: e.g., [30, 23, 40]
        -> [30] and [23, 40]
        -> [30] and [23 * exp(1) + 40 * exp(2) = 358.0827]
        -> [30] and [358.0827 scaled to 0.3580827]
        ->  30 + 0.3580827 = 30.3580827
        the integer part is the final token vocab id
        '''
        linear_hash_weight = np.array([0] + [math.exp(i+1) for i in range(n_of_n_gram - 1)])
        general_vocab_ids_4_gram_hash = (general_vocab_ids_4_gram[:, :n_of_n_gram] @ linear_hash_weight[:, None])[:, 0]
        general_vocab_ids_4_gram_hash = general_vocab_ids_4_gram_hash / \
                                        np.power(np.log10(general_vocab_ids_4_gram_hash + 1.) + 1, 10)
        n_gram = general_vocab_ids_4_gram_hash + general_vocab_ids_4_gram[:, 0]
        del general_vocab_ids_4_gram_hash
        
        '''
        slow solution
        n_gram: e.g., [30, 23, 40] -> '30-23-40', the first element is the final token vocab id
        n_gram = ['.'.join([str(w) for w in grm_n]]) for grm_n in general_vocab_ids_4_gram[:,:n_of_n_gram]]
        '''
    else:
        raise NotImplementedError("not implemented for n = %d" % n_of_n_gram) 
    logger.info('N-gram ppl established.')

    # --- get n-gram table
    '''slow solution
    table_n_gram = set(n_gram)
    table_n_gram_idx_dict = dict(zip(table_n_gram, [[] for k in table_n_gram]))
    for idxs, gram in enumerate(tqdm(n_gram)):
        table_n_gram_idx_dict[gram].append(idxs)
    '''
    # fast solution
    logger.info('table_n_gram set establishing...')
    table_n_gram_counter = Counter(n_gram)
    # table_n_gram_counter = Counter(dict((k, v) for k, v in tqdm(table_n_gram_counter.items()) if v >= mininum_sample))
    table_n_gram = list(table_n_gram_counter.keys())
    logger.info('%d table_n_gram set established.' % len(table_n_gram))
    
    # ---
    logger.info('table_n_gram_idx_dict initialization...')
    table_n_gram_idx_dict = {}
    for k in tqdm(table_n_gram):
        table_n_gram_idx_dict[k] = np.zeros(table_n_gram_counter[k], dtype=np.int64)
    logger.info('table_n_gram_idx_dict initialized.')

    # ---
    logger.info('%d way N-gram table dict establishing...' % len(table_n_gram))
    for idx, gram in enumerate(tqdm(n_gram)):
        if table_n_gram_counter[gram] <= 0:
            continue
        table_n_gram_counter[gram] -= 1
        table_n_gram_idx_dict[gram][table_n_gram_counter[gram]] = idx
    del table_n_gram_counter
    logger.info('%d way N-gram table dict established.' % len(table_n_gram))
    '''
    NOTE: about table_n_gram_idx_dict
    For a trainset that contains 6 sentences:
        I:   'this is a good place'
        II:  'it is rainy.'
        III: 'he is good'
        IV:  'i think he is excellent'
        V:   'yes he is'
        VI:  'is it ?'

    We build the datastore:
    0-this, 1-is, 2-a, 3-good, 4-place,
    5-it, 6-is, 7-rainy,
    8-he, 9-is, 10-good,
    11-i, 12-think, 13-he, 14-is, 15-excellent,
    16-yes, 17-he, 18-is,
    19-is, 20-it, 21-?

    the 1-gram list of "is":  [
        [1('this is')],
        [6('it is')],
        [9('he is')],
        [14('he is')],
        [18('he is')],
        [19('is')]
    ]

    the 2-gram list of "is" that ends with the token "is": [
        [1 ('this is')],
        [6 ('it is')],
        [9, 14, 18 ('he is')],
        [19 ('[padding] is')]
    ]
    etc.
    '''


    # --- start pruning
    logger.info('Start %s pruning...' % prune_style)

    thread_num = 30
    thread_width = len(table_n_gram_idx_dict) // thread_num + 1
    pool = Pool(processes=thread_num)

    table_n_gram_idx_dict_keys = list(table_n_gram_idx_dict.keys())
    logger.info('Pruning threads start.')

    results = [pool.apply_async(
            func=n_gram_prune_thread_inner_table_n_gram_idx_dict,
            args=(
                dict([(k, table_n_gram_idx_dict[k]) for k in \
                    table_n_gram_idx_dict_keys[i*thread_width:min((i+1)*thread_width, len(table_n_gram_idx_dict))]]),
                prune_style, 
                mininum_sample, 
                sample_rate,
                n_gram_uniform_ppl if 'ppl' in prune_style else None,
                tgt_entropy if 'entropy' in prune_style else None
                ),
            ) for i in range(thread_num)]
    pool.close(), pool.join()
    table_n_gram_idx_dict = {}
    for res in results: table_n_gram_idx_dict.update(res.get())
    table_n_gram_idx_dict_keys = list(table_n_gram_idx_dict.keys())
    logger.info('Pruning threads done.')
 


    # --- collection of pruned results
    logger.info('Start collecting pruned n-grams.')

    thread_num = 30
    pool = Pool(processes=thread_num)
    thread_width = len(table_n_gram_idx_dict) // thread_num + 1

    logger.info('Collection threads start.')
    results = [pool.apply_async(
            func=collect_pruned_n_grams_thread,
            args=(
                dict([(k, table_n_gram_idx_dict[k]) \
                    for k in table_n_gram_idx_dict_keys[i*thread_width:min((i+1)*thread_width, len(table_n_gram_idx_dict))]]),
                ),
            ) for i in range(thread_num)
    ]
    pool.close(), pool.join()
    logger.info('Collection threads done.')


    val_list, dbidx_list, key_list, tgt_lens_list, src_lens_list = [], [], [], [], []
    for res in tqdm(results):
        val_l, dbidx_l, tgt_lens_l, src_lens_l = res.get()
        val_list.extend(val_l)
        dbidx_list.extend(dbidx_l)
        key_list.extend([[general_keys[k] for k in np_idxs] for np_idxs in dbidx_l])
        # tgt_lens_list.extend(tgt_lens_l)
        # src_lens_list.extend(src_lens_l)
    logger.info('### Clustering is done, we pruned %f of the datastore, getting %d n-gram, %d nodes from %d nodes' % (
        sum([len(keys) for keys in key_list]) / general_dstore_size,
        len(key_list), sum([len(keys) for keys in key_list]), general_dstore_size))
    logger.info('### N-gram pruning done, took {} s'.format(time.time() - start))
    return key_list, val_list, dbidx_list, None, None #tgt_lens_list, src_lens_list


def cluster_and_prune(
    general_dstore_size: int,
    general_keys: np.array,
    general_vocab_ids: np.array,
    general_tgt_lens: np.array,
    general_src_lens: np.array,
    dictionary: List[str],
    cluster_params: Dict
    ):

    ## frequence collection
    key_list = [[] for _ in range(len(dictionary))]
    dbidx_list = [[] for _ in range(len(dictionary))]
    tgt_lens_list = [[] for _ in range(len(dictionary))]
    src_lens_list = [[] for _ in range(len(dictionary))]
    # for i in tqdm(range(general_dstore_size // 10)): # for debugging
    for dbidx in tqdm(range(general_dstore_size)):
        vocab_id = general_vocab_ids[dbidx][0]
        key_list[vocab_id].append(general_keys[dbidx])
        dbidx_list[vocab_id].append(dbidx)
        tgt_lens_list[vocab_id].append(general_tgt_lens[dbidx])
        src_lens_list[vocab_id].append(general_src_lens[dbidx])

    del general_vocab_ids
    del general_keys

    ## general datastore cluster
    cluster_type = 'birch'
    min_samples = cluster_params['min_samples']
    eps = cluster_params['eps']
    threshold = cluster_params['threshold']
    cluster_algorithm_list = ['spectrum', 'dbscan', 'birch']
    assert cluster_type in cluster_algorithm_list, \
        'the cluster algorithm should be in the list: ' + ' '.join(cluster_algorithm_list)
    
    if cluster_type == 'spectrum':
        sc = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', n_init=3, n_neighbors=min_samples)
    elif cluster_type == 'dbscan':
        sc = DBSCAN(eps=eps, min_samples=min_samples)
    elif cluster_type == 'birch':
        sc = Birch(n_clusters=None)
    
    logger.info('### Start clustering ...')
    new_key_list = []
    new_val_list = []
    new_dbidx_list = []
    new_tgt_lens_list = []
    new_src_lens_list = []
    base_number = min_samples
    sample_bound = 100000000
    for vocab_id, keys_dbidx in enumerate(zip(key_list, dbidx_list, tgt_lens_list, src_lens_list)):
        if vocab_id % 1000 == 0:
            logger.info('clustering %d-th vocab...' % vocab_id)

        keys, dbidx, tgt_len, src_len = keys_dbidx[0], keys_dbidx[1], keys_dbidx[2], keys_dbidx[3]

        if len(keys) == 0:
            continue

        '''
        key_list[0] is a list of all-zero keys, because vocab[0] is '<s>'
        key_list[1~3] are not all-zero keys, of which the vocabs are '<pad> </s> <unk>'
        '''
        if vocab_id < 4 and vocab_id != 2:
            continue

        if len(keys) < base_number:
            new_key_list.append(keys)
            new_dbidx_list.append(dbidx)
            new_tgt_lens_list.append(tgt_len)
            new_src_lens_list.append(src_len)
            new_val_list.append([vocab_id for _ in range(len(keys))])
            continue 

        # --- to decrease the computation, we just sample some nodes of the same vocab_id
        if len(keys) > sample_bound:
            keys_dbidx = random_sample(keys_dbidx, sample_bound)
            keys, dbidx = [item[0] for item in keys_dbidx], [item[1] for item in keys_dbidx]
            tgt_len, src_len = [item[2] for item in keys_dbidx], [item[3] for item in keys_dbidx]

        # --- turn list of np.array into a stack np.array
        keys = np.array(keys).astype(np.float32)

        # --- start cluster
        if use_cluster:
            if cluster_type == 'birch':
                sc = Birch(n_clusters=None, threshold=threshold)#, branching_factor=256)
                clustering = sc.fit(keys / 10.) # train
                labels = clustering.labels_
                sc.partial_fit() # global clustering, refine
                labels = sc.predict(keys / 10.) 
            else:
                clustering = sc.fit(keys)
            # sc.n_clusters = 0 # int(math.log(len(keys)+base_number, base_number+1))
            # sc.n_neighbors = min(len(keys), min_samples)
        else:
            labels = np.zeros(keys.shape[0], dtype=np.int)

        tmp_key = [[] for _ in range(labels.max() + 1)]
        tmp_dbidx = [[] for _ in range(labels.max() + 1)]
        tmp_tgt_len = [[] for _ in range(labels.max() + 1)]
        tmp_src_len = [[] for _ in range(labels.max() + 1)]
        for n in range(labels.shape[0]):
            if labels[n] == -1: ## isolated node
                continue
            tmp_key[labels[n]].append(keys[n])
            tmp_dbidx[labels[n]].append(dbidx[n])
            tmp_tgt_len[labels[n]].append(tgt_len[n])
            tmp_src_len[labels[n]].append(src_len[n])

        tmp_key = [keys for keys in tmp_key if len(keys) != 0]
        # tmp_key = [random_sample(keys) for keys in tmp_key]
        new_key_list.extend(tmp_key)
        
        tmp_dbidx = [dbidx for dbidx in tmp_dbidx if len(dbidx) != 0]
        new_dbidx_list.extend(tmp_dbidx)

        tmp_tgt_len = [l for l in tmp_tgt_len if len(l) != 0]
        new_tgt_lens_list.extend(tmp_tgt_len)

        tmp_src_len = [l for l in tmp_src_len if len(l) != 0]
        new_src_lens_list.extend(tmp_src_len)

        tmp_val = [[vocab_id for _ in range(len(key))] for key in tmp_key]
        new_val_list.extend(tmp_val)
        assert len(tmp_key) == len(tmp_val)
        assert len(tmp_key) == len(tmp_dbidx)

    key_list = new_key_list
    val_list = new_val_list
    dbidx_list = new_dbidx_list
    tgt_lens_list = new_tgt_lens_list
    src_lens_list = new_src_lens_list
    '''
    After target-side clustering, tokens of the same vocab may be split
    into different slices of this new_val_list, like:
    new_val_list = 
    [
        ...
        [5,5,5], [5,5,5,5,5],
        [6,], [6,6,6,6], [6,6,6], [6,6],
        [7],
        [8,8,8,8], [8,8],
        ...
    ]
    '''
    logger.info('### Clustering is done, we get %d clusters, %d nodes' % (
        len(key_list), sum([len(keys) for keys in key_list]) ))
    return key_list, val_list, dbidx_list, tgt_lens_list, src_lens_list


def pruned_by_probability(
        keys : np.array,
        dbidxs : np.array,
        key_center : np.array,
        tgt_lens : np.array,
        src_lens : np.array,
        threshold : float = 0.8,
    ):
    nums = dbidxs.shape[0]
    assert dbidxs.shape[0] == keys.shape[0]
    if nums < 16:
        return keys, dbidxs, tgt_lens, src_lens

    # --- mask type 1
    distance = np.linalg.norm(keys - key_center[None], ord=2, axis=1)
    selected_num = max(int(distance.shape[0] * threshold), 1)
    mask = np.argpartition(-distance, -selected_num)[-selected_num:] # get max-k

    # --- mask type 2
    # distance = np.linalg.norm(keys - key_center[None], ord=2, axis=1)
    # probabilities = np.exp(-distance)
    # mask = probabilities >= threshold

    if mask.sum() == 0:
        argmax = np.argmax(-distance)
        keys = keys[argmax:argmax+1]
        dbidxs = dbidxs[argmax:argmax+1]
        tgt_lens = tgt_lens[argmax:argmax+1]
        src_lens = src_lens[argmax:argmax+1]
    else:
        keys = keys[mask]
        dbidxs = dbidxs[mask]
        tgt_lens = tgt_lens[mask]
        src_lens = src_lens[mask]
    # print(distance, mask.sum(), keys.shape, dbidxs.shape, probabilities)
    # logger.info('filter %d keys' % nums-dbidxs.shape[0])
    return keys, dbidxs, tgt_lens, src_lens


def main(paradict: Dict):
    dstore_filenames = paradict['dstore_filenames']
    decoder_knn_compact_dim = paradict['decoder_knn_compact_dim']
    dstore_sizes = paradict['dstore_sizes']
    general_dstore_size = sum(paradict['general_dstore_sizes'])
    general_dstore_sizes = paradict['general_dstore_sizes']
    dstore_fp16 = paradict['dstore_fp16']
    domains = paradict['domains']
    general_dstore_filenames = paradict['general_dstore_filenames']
    verbose = paradict['verbose']
    pruned_dstore_filenames = paradict['pruned_dstore_filenames']
    cluster_params = paradict['dbscan']
    prune_style = paradict['prune_style']
    sample_rate = paradict['sample_rate']

    # --- get vocab
    with open('data-bin/%s/dict.de.txt' % domains[0]) as f:
        dictionary = ['<s>', '<pad>', '</s>', '<unk>'] + [s.strip().split()[0] for s in f.readlines()]

    # --- info of valid datastore
    # [it]="33538" [medical]="56614" [koran]="58319" [law]="82352
    # for dstore_size, dstore_filename, domain in zip(dstore_sizes, dstore_filenames, domains):
    #     valid_keys, valid_vocab_ids = get_mt_datastore(
    #         dstore_filename,
    #         dstore_fp16,
    #         dstore_size,
    #         decoder_knn_compact_dim,
    #         mode='r')
    #     key_list = [[] for _ in dictionary]
    #     for i in range(dstore_size):
    #         key_list[valid_vocab_ids[i][0]].append(valid_keys[i])
    #     for i in range(len(dictionary)):
    #         logger.info('%s vocabulary-%s has %d keys' % (dstore_filename, dictionary[i], len(key_list[i])))
    #     distribution = [len(kl) for kl in key_list]
    #     non_empty_num = sum([1 for d in distribution if d > 0])
    #     draw_vocab_distribution(dictionary, distribution, domain)


    # -- start pruning 
    if len(general_dstore_sizes) != 1 and general_dstore_sizes[0] == general_dstore_sizes[-1]:
        read_dstore_size = 0

    for domain, dstore_size, dstore_filename, domain_true_dstore_size, general_dstore_filename, pruned_dstore_filename \
            in zip(domains, dstore_sizes, dstore_filenames, general_dstore_sizes,
                general_dstore_filenames, pruned_dstore_filenames):

        if len(general_dstore_sizes) != 1 and general_dstore_sizes[0] != general_dstore_sizes[-1]:
            read_dstore_size = 0

        # --- get a general datastore
        logger.info('Reading general datastore...')
        general_keys, general_vocab_ids, general_tgt_lens, general_src_lens, \
            general_vocab_ids_4_gram, general_tgt_4_gram_probs, general_tgt_entropy = get_mt_datastore(
                general_dstore_filename,
                dstore_fp16,
                domain_true_dstore_size,
                decoder_knn_compact_dim,
                mode='r')

        # --- n-gram cluster and prune
        key_list, val_list, dbidx_list, tgt_lens_list, src_lens_list = n_gram_prune(
            domain_true_dstore_size,
            general_keys,
            general_vocab_ids_4_gram,
            general_tgt_lens,
            general_src_lens,
            general_tgt_4_gram_probs,
            general_tgt_entropy,
            dictionary,
            sample_rate=sample_rate,
            prune_style=prune_style
        )
        
        if not use_valset_to_retrieve:
            # --- save as a new pruned datastore
            if os.path.exists(pruned_dstore_filename):
                shutil.rmtree(pruned_dstore_filename)
            os.makedirs(pruned_dstore_filename, exist_ok=True)
            pruned_dstore_keys, pruned_dstore_vocab_ids, _, _, _, _, _ = get_mt_datastore(
                pruned_dstore_filename,
                dstore_fp16,
                sum([len(keys) for keys in key_list]),
                decoder_knn_compact_dim,
                mode='w+')

            n = sum([len(v) for v in val_list])

            pruned_dstore_keys[:n] = np.vstack([np.vstack(v) for v in key_list]).astype(np.float16 if dstore_fp16 else np.float32)
            pruned_dstore_vocab_ids[:n] = np.vstack([np.vstack(v) for v in val_list])
            del key_list, val_list
            logger.info('%d\n\n' % n)

            with open(pruned_dstore_filename + '/%d' % n, 'w') as f:
                f.close()
            continue

        # --- retrieval and merge using a given valset datastore (usually small)
        '''
        NOTE: the rest part of the main function was duplicated for PCKMT.
        '''
        # logger.info('Clustering general datastore...')
        # key_list, val_list, dbidx_list, tgt_lens_list, src_lens_list = cluster_and_prune(
        #     general_dstore_size,
        #     general_keys,
        #     general_vocab_ids,
        #     general_tgt_lens,
        #     general_src_lens,
        #     dictionary,
        #     cluster_params)

        # # --- debug retrieval w.r.t. valid w/o clustering
        # key_list = [[general_keys[i]] for i in range(general_dstore_size)]
        # val_list = general_vocab_ids.tolist()

        # --- collect general cluster info
        key_list_mean = [None] * len(key_list)
        for cluster_id, keys in enumerate(key_list):
            key_list_mean[cluster_id] = np.vstack(keys).astype(np.float32).mean(axis=0)

            # --- all elements in val_list[cluster_id] are the same, so reduce them.
            val_list[cluster_id] = val_list[cluster_id][0] 

        # --- debug: get cluster distances inner the general datastore
        # key_list_mean_array = np.vstack(key_list_mean) # (k_cluster, dim)
        # k_k_distance = np.linalg.norm(key_list_mean_array[None] - key_list_mean_array[:, None], ord=2, axis=-1)
        # logger.debug(k_k_distance.mean(-1).mean(-1))

        original_key_list_mean_collection = [[] for _ in dictionary]
        original_tgt_lens_list_mean_collection = [[] for _ in dictionary]
        original_key_list_collection = [[] for _ in dictionary]
        original_dbidx_list_collection = [[] for _ in dictionary]
        original_val_list_collection = [[] for _ in dictionary]
        original_tgt_lens_list_collection = [[] for _ in dictionary]
        original_src_lens_list_collection = [[] for _ in dictionary]
        for cluster_id, vocab_id in enumerate(val_list):
            original_key_list_mean_collection[vocab_id].append(key_list_mean[cluster_id])
            original_tgt_lens_list_mean_collection[vocab_id].append(np.vstack(tgt_lens_list[cluster_id]).mean(0))
            original_key_list_collection[vocab_id].append(np.vstack(key_list[cluster_id]))
            original_dbidx_list_collection[vocab_id].append(np.vstack(dbidx_list[cluster_id]))
            original_val_list_collection[vocab_id].append(cluster_id)
            original_tgt_lens_list_collection[vocab_id].append(np.vstack(tgt_lens_list[cluster_id]))
            original_src_lens_list_collection[vocab_id].append(np.vstack(src_lens_list[cluster_id]))

        # logger.debug(sum([1 for key in key_list_mean_collection if len(key) != 0]))
        '''
        @val_list_collection:
        [
            vocab_0: [10, 11, 13],
            vocab_1: [1, 3, 4, 5, 6],
            vocab_2: [2, 5, 6],
            ...
        ]
        @key_list_mean_collection:
        collects the correspoinding mean vectors of each cluster 
        and then assign them to corresponding vocab_id buckets
        '''
        original_key_list_mean_collection = [np.vstack(keys) if len(keys) else None \
                                                for keys in original_key_list_mean_collection]
        original_tgt_lens_list_mean_collection = [np.vstack(keys) if len(keys) else None \
                                                for keys in original_tgt_lens_list_mean_collection]

        del key_list
        del val_list
        del dbidx_list


        # --- initilize
        key_list_mean_collection = deepcopy(original_key_list_mean_collection)
        tgt_lens_list_mean_collection = deepcopy(original_tgt_lens_list_mean_collection)
        key_list_collection = deepcopy(original_key_list_collection)
        dbidx_list_collection = deepcopy(original_dbidx_list_collection)
        val_list_collection = deepcopy(original_val_list_collection)
        tgt_lens_list_collection = deepcopy(original_tgt_lens_list_collection)
        src_lens_list_collection = deepcopy(original_src_lens_list_collection)

        logger.info('### Starting retrieval on %s domain ...' % domain)

        if general_dstore_size < dstore_size:
            logger.warning(
                'the size of the given general dstore is too small. \n \
                general_size vs target_size = %d vs %d' % (general_dstore_size, dstore_size))

        # --- get validate datastore
        valid_keys, valid_vocab_ids, valid_tgt_lens, valid_src_lens, valid_vocab_ids_4_gram, _, _ = get_mt_datastore(
            dstore_filename,
            dstore_fp16,
            dstore_size,
            decoder_knn_compact_dim,
            mode='r')

        # --- which set of vocab does not appear in valid set
        not_appeared_vocab_list_in_valid_set = [True for _ in dictionary]
        # for i in range(4):
        #     not_appeared_vocab_list_in_valid_set[i] = False
        for i in range(dstore_size):
            not_appeared_vocab_list_in_valid_set[valid_vocab_ids[i][0]] = False
        # not_appeared_vocab_list_in_valid_set = [False for _ in dictionary]
        
        logger.info('%s: %d vocabs not in the valid set.' % (domain, sum(not_appeared_vocab_list_in_valid_set)))

        # --- compute similarity scores of datastore clusters w.r.t. the valset
        val_select_flag_list = [[0] * len(cluster_ids) for cluster_ids in val_list_collection]
        moving_score = 0.
        moving_step = 0
        for i in tqdm(range(dstore_size)):            
            target_vocab_id = valid_vocab_ids[i][0]
            key_mean_collection_slice = key_list_mean_collection[target_vocab_id]
            tgt_lens_list_mean_collection_slice = tgt_lens_list_mean_collection[target_vocab_id]
            
            if key_mean_collection_slice is None:
                continue

            query = valid_keys[i]
            query_len = valid_tgt_lens[i]

            ## dot score
            # score = query[None] @ key_mean_collection_slice.T # 1 * m @ m * k
            
            # ## l2 score
            # dist = query[None] - key_mean_collection_slice
            # score = -(dist * dist).sum(-1)

            ## tgt lens distance
            dist = np.abs(query_len[None] - tgt_lens_list_mean_collection_slice)
            # dist = dist / tgt_lens_list_mean_collection_slice
            score = -dist

            # logger.debug(score)

            # --- filter unreliable retrieval
            maxarg_score = np.argmax(score)
            score_threshold = valset_similarity_threshold
            if maxarg_score < score_threshold:
                continue

            for score_id in range(score.shape[-1]):
                if val_select_flag_list[target_vocab_id][score_id] == 1: # already selected
                    continue

                if score[score_id] < score_threshold:
                    continue

                val_select_flag_list[target_vocab_id][score_id] = 1

                # --- update clustering score
                moving_score = moving_score * moving_step / float(moving_step + 1) + \
                                score[score_id] * 1. / float(moving_step + 1)
                moving_step += 1

        # --- gmm pruned on seen vocabs, low score clusters
        logger.info('### %s domain pruned on seen vocab is started, we get %d nodes' % (domain, \
                    sum([sum([key.shape[0] for key in keys_list]) for keys_list in key_list_collection])))

        if gmm_pruned_on_seen_vocabs:
            for target_vocab_id in range(len(dictionary)):
                if not not_appeared_vocab_list_in_valid_set[target_vocab_id]:
                    # --- search for seen tokens
                    for idx in range(len(key_list_collection[target_vocab_id])):
                        if val_select_flag_list[target_vocab_id][idx] == 0: # low score clusters
                            pruned_out = pruned_by_probability(
                                keys=key_list_collection[target_vocab_id][idx],
                                dbidxs=dbidx_list_collection[target_vocab_id][idx],
                                key_center=key_list_mean_collection[target_vocab_id][idx],
                                tgt_lens=tgt_lens_list_collection[target_vocab_id][idx],
                                src_lens=src_lens_list_collection[target_vocab_id][idx],
                                threshold=1.0,
                            )
                            key_list_collection[target_vocab_id][idx] = pruned_out[0]
                            dbidx_list_collection[target_vocab_id][idx] = pruned_out[1]
                            tgt_lens_list_collection[target_vocab_id][idx] = pruned_out[2]
                            src_lens_list_collection[target_vocab_id][idx] = pruned_out[3]
                        val_select_flag_list[target_vocab_id][idx] = 1
        logger.info('### %s domain pruned on seen vocab is done, we get %d nodes' % (domain, 
                    sum([sum([key.shape[0] for key in keys_list]) for keys_list in key_list_collection])))

        # --- prune unseen vocab, all clusters
        logger.info('### %s domain pruned on unseen vocab is started, we get %d nodes' % (domain, 
                    sum([sum([key.shape[0] for key in keys_list]) for keys_list in key_list_collection])))

        if gmm_pruned_on_unseen_vocabs:
            for target_vocab_id in range(len(dictionary)):
                if not_appeared_vocab_list_in_valid_set[target_vocab_id]:
                    for idx in range(len(key_list_collection[target_vocab_id])):
                        assert val_select_flag_list[target_vocab_id][idx] == 0
                        pruned_out = pruned_by_probability(
                            keys=key_list_collection[target_vocab_id][idx],
                            dbidxs=dbidx_list_collection[target_vocab_id][idx],
                            key_center=key_list_mean_collection[target_vocab_id][idx],
                            tgt_lens=tgt_lens_list_collection[target_vocab_id][idx],
                            src_lens=src_lens_list_collection[target_vocab_id][idx],
                            threshold=1.0
                        )
                        key_list_collection[target_vocab_id][idx] = pruned_out[0]
                        dbidx_list_collection[target_vocab_id][idx] = pruned_out[1]
                        tgt_lens_list_collection[target_vocab_id][idx] = pruned_out[2]
                        src_lens_list_collection[target_vocab_id][idx] = pruned_out[3]
                        val_select_flag_list[target_vocab_id][idx] = 1
        logger.info('### %s domain pruned on unseen vocab is done, we get %d nodes' % (domain, 
                sum([sum([key.shape[0] for key in keys_list]) for keys_list in key_list_collection])))

        # --- filter clusters that are not in the retrieval results
        all_select_keys, all_select_vocab_ids, all_select_dbidx = [], [], []
        for vocab_id in range(len(dictionary)):
            select_flag = val_select_flag_list[vocab_id]
            if not not_appeared_vocab_list_in_valid_set[vocab_id]:
                select_flag = [1 for _ in select_flag]

            if sum(select_flag) == 0:
                continue

            keys = key_list_collection[vocab_id]
            dbidx = dbidx_list_collection[vocab_id]

            select_keys = [keys[i] for i, select_or_not in enumerate(select_flag) if select_or_not == 1]
            select_dbidx = [dbidx[i] for i, select_or_not in enumerate(select_flag) if select_or_not == 1]

            # mean_keys = key_list_mean_collection[vocab_id]
            # select_keys = [keys[i] if select_or_not == 1 else mean_keys[i] for i, select_or_not in enumerate(select_flag)]

            if select_keys:
                stack_select_keys = np.vstack(select_keys)
                all_select_keys.extend(stack_select_keys)
                all_select_vocab_ids.extend(np.full((stack_select_keys.shape[0], 1), vocab_id))
                all_select_dbidx.extend(np.vstack(select_dbidx))
                # logger.info('vocab-%d: %s -> %d clusters, %d nodes, not appeared in valid: %d' % (
                #     vocab_id,
                #     dictionary[vocab_id],
                #     len(select_keys),
                #     stack_select_keys.shape[0],
                #     not_appeared_vocab_list_in_valid_set[vocab_id])
                # )

            assert len(select_flag) == len(keys), (len(select_flag), len(keys))


        if len(all_select_keys) == 0:
            logger.warning('%s domain is too far from general domain (no acceptable retrieval)\n\n' % domain)
            continue

        all_select_keys = np.vstack(all_select_keys)
        all_select_vocab_ids = np.vstack(all_select_vocab_ids)
        all_select_dbidx = np.vstack(all_select_dbidx)

        logger.info('after pruning, get %d clusters, get %s nodes, the average loss is %f.' % (
            sum([sum(v) for v in val_select_flag_list]),
            all_select_keys.shape,
            moving_score))

        # --- save new pruned datastore
        pruned_dstore_keys, pruned_dstore_vocab_ids, _, _, _, _, _ = get_mt_datastore(
            pruned_dstore_filename,
            dstore_fp16,
            all_select_keys.shape[0],
            decoder_knn_compact_dim,
            mode='w+')
        pruned_dstore_keys[:] = all_select_keys.astype(np.float16 if dstore_fp16 else np.float32)
        pruned_dstore_vocab_ids[:] = all_select_vocab_ids

        # --- cluster efficiency 
        domain_prediction = np.memmap(
            pruned_dstore_filename + '/selection.npy',
            dtype=np.bool,
            mode='w+',
            shape=(general_dstore_size))
        domain_prediction[all_select_dbidx] = True

        domain_label = np.zeros((general_dstore_size), np.int)
        domain_label[read_dstore_size:read_dstore_size+domain_true_dstore_size] = 1

        precision = precision_score(domain_label, domain_prediction) # tp / (tp + fp)
        recall = recall_score(domain_label, domain_prediction) # tp / (tp + fn)

        read_dstore_size += domain_true_dstore_size
        logger.info('Cluster precision: %f\tRecall: %f' % (precision, recall))

        del all_select_keys
        del all_select_vocab_ids
        del all_select_dbidx
    logger.info('Saving done.\n')


def cli_main():
    prune_style = 'random'
    unsupervise_prune_sample_rate = 0.9

    dstore_fp16 = True
    verbose = True
    decoder_knn_compact_dim = 64
    dstore_infos = {
        "medical": [56614, 6903142],
        "it": [33538, 3602863],
        "koran": [58319, 524375],
        "law": [82352, 19061383],
        "subtitles": [25036, 153604142],
    }
    dstore_domains, dstore_domains_sizes = zip(*(dstore_infos.items()))
    dstore_sizes, general_dstore_sizes = zip(*dstore_domains_sizes)

    dstore_filenames = ['save_datastore/%s/knn_transfered_nce_valid' % s for s in dstore_domains]
    general_dstore_filenames = ['save_datastore/%s/knn_transfered_nce' % s for s in dstore_domains]


    pruned_dstore_filenames = ['save_datastore/merge/knn_transfered_nce/pruned+seen_pruned+unseen_pruned/%s/%s-%.2f' \
        % (s, prune_style, unsupervise_prune_sample_rate) for s in dstore_domains]

    paradict = {
        'dstore_sizes': dstore_sizes,
        'general_dstore_sizes': general_dstore_sizes,
        'dstore_filenames': dstore_filenames,
        'decoder_knn_compact_dim': decoder_knn_compact_dim,
        'dstore_fp16': dstore_fp16,
        'domains': dstore_domains,
        'general_dstore_filenames': general_dstore_filenames,
        'verbose': verbose,
        'pruned_dstore_filenames': pruned_dstore_filenames,
        # --- pruned method
        'sample_rate': unsupervise_prune_sample_rate,
        'prune_style': prune_style,
        # --- dbscan cluster part (NOTE: BIRCH is more practical for large datastore)
        'dbscan': {'min_samples':4, 'eps': 10, 'threshold': 0.5},
        # 'dbscan': {'min_samples': 4, 'eps': 1} 1640080 nodes 28227 cluster
        # 'dbscan': {'min_samples': 4, 'eps': 2} 3986843 nodes 37962 clusters
        # 'dbscan': {'min_samples': 4, 'eps': 4} 7947795 nodes 23823 clusters
        # 'dbscan': {'min_samples': 4, 'eps': 10} 9179043 nodes 17833 clusters
    }
    main(paradict)    
 

if __name__ == "__main__":
    cli_main()