# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from os import remove
from typing import Any, Dict, List, Optional, Tuple
import time

from tqdm import tqdm
import numpy as np
from random import randint, sample

import torch 
import torch.nn as nn
from fairseq import utils 
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from fairseq.modules.knn_datastore import KNN_Dstore
import torch.nn.functional as functional
from torch_scatter import scatter
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from torch.nn.parameter import Parameter

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

from torch.utils.data import Dataset, DataLoader

# --- 
# NOTE: Pytorch api applies workers to parallel treat the dataloader while ours is not.
# Another difference between pytorch dataloader api is that pytorch api is not bagging-style while ours is.
PYTORCH_DATAAPI = False
min_samples = 4

class TripletDatastoreSamplingDataset(Dataset):
    def __init__(self,
            args,
            dictionary = None,
            use_cluster : bool = False,
            cluster_type : str = 'dbscan',
            verbose = False,
            knn_datastore = None,
            sample_rate: float = 0.4
        ):
        assert knn_datastore is not None
        assert sample_rate <= 1.0

        db_keys, db_vals = knn_datastore.keys, knn_datastore.vals

        if sample_rate != 1.0:
            random_sample = np.random.choice(np.arange(db_vals.shape[0]), size=sample_rate * db_vals.shape[0], replace=False)
            db_keys = db_keys[random_sample]
            db_vals = db_vals[random_sample]

        vocab_freq = [0 for _ in range(len(dictionary))]
        key_list = [[] for _ in range(len(dictionary))]

        # ## (deprecated) filter vocabulary from unrelevant languages
        # import langid
        # unwanted_language = ['zh', 'ko', 'ja']
        # wanted_vocab = [True for _ in range(len(dictionary))]
        # if verbose: print('unwanted vocabularies are = ')
        # for i, voc in enumerate(dictionary.symbols):
        #     voc = voc.replace('@', '') # remove bpe symbols
        #     if langid.classify(voc)[0] in unwanted_language:
        #         if verbose: print(voc, end='')
        #         wanted_vocab[i] = False
        # if verbose:
        #     print('\n total number of dictionary = %d' % len(dictionary))
        #     print("the number of unwanted vocabularies = %d, almost %f of all" % \
        #           (len(dictionary) - sum(wanted_vocab), (len(dictionary) - sum(wanted_vocab)) / len(dictionary)) )

        ## frequence collection
        for i in tqdm(range(args.dstore_size)):
            val = db_vals[i]
            vocab_freq[val] += 1
            key_list[val].append(db_keys[i])

        del db_vals
        del db_keys
        del knn_datastore

        if use_cluster:
            ## inner clustering refine
            cluster_algorithm_list = ['spectrum', 'dbscan']
            assert cluster_type in cluster_algorithm_list, 'the cluster algorithm should be in the list: ' + ' '.join(cluster_algorithm_list)
            
            if cluster_type == 'spectrum':
                from sklearn.cluster import SpectralClustering
                sc = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', n_init=3, n_neighbors=min_samples)
            elif cluster_type == 'dbscan':
                from sklearn.cluster import DBSCAN
                sc = DBSCAN(eps=10, min_samples=min_samples)
            
            print('start clustering ...')
            new_key_list = []
            new_val_list = []
            base_number = min_samples
            # Limited by memory, 100000 koran/it/medical (<=10M) 20000 for law/subtitles (>=19M). 
            sample_bound = 100000
            for vocab_id, keys in tqdm(enumerate(key_list)):
                if len(keys) == 0:
                    continue

                if vocab_id % 2000 == 0:
                    print('clustering %d' % vocab_id)

                '''
                key_list[0] is a list of all-zero keys, because vocab[0] is '<s>'
                key_list[1~3] are not all-zero keys, of which the vocabs are '<pad> </s> <unk>'
                '''
                if vocab_id < 4 and vocab_id != 2:
                    continue

                if len(keys) <= base_number:
                    new_key_list.append(keys)
                    new_val_list.append([vocab_id for _ in range(len(keys))])
                    continue

                ## to decrease the computation
                if len(keys) > sample_bound:
                    keys = sample(keys, sample_bound)

                sc.n_clusters = int(math.log(len(keys)+base_number, base_number))
                sc.n_neighbors = min(len(keys), min_samples)

                keys = np.array(keys)

                clustering = sc.fit(keys)
                labels = clustering.labels_

                tmp_key = [[] for _ in range(labels.max()+1)]
                for n in range(labels.shape[0]):
                    if labels[n] == -1:
                        continue
                    tmp_key[labels[n]].append(keys[n])
                    # print(labels[j], end=' ')
                tmp_key = [key for key in tmp_key if len(key) != 0]
                new_key_list.extend(tmp_key)

                tmp_val = [[vocab_id for _ in range(len(key))] for key in tmp_key]
                new_val_list.extend(tmp_val)
                assert len(tmp_key) == len(tmp_val)

            del key_list
            self.key_list = new_key_list
            self.val_list = new_val_list
            '''
            After target-side clustering, tokens of the same vocab may be split
            into different slices of this new_val_list, like:
            [
             [5,5,5], [5,5,5,5,5],
             [6,], [6,6,6,6], [6,6,6], [6,6],
             [7],
             [8,8,8,8], [8,8],
              ...
            ]
            '''

            print('we get %d clusters' % len(self.key_list))

            # # post-processing
            # for i in range(len(self.key_list)):
            #     if len(self.key_list[i]) == 0:
            #         continue
            #     self.key_list[i] = np.array(self.key_list[i])

            print('cluster done. Get %d nodes' % sum([len(keys) for keys in self.key_list]))

        ## (deprecated)
        # if verbose: print('==== wanted vocabularies\' frequence ====')
        # with open('val_list_it', 'w') as f:
        #     for d, v, wv in zip(dictionary.symbols, vocab_freq, wanted_vocab):
        #         f.write(str(v) + '\n')
        #         if wv and verbose:
        #             print(d, v, end='  ')

        ## statistics collection of vocab frequency
        self.larger_than_2_vocab  = [i for i, v in enumerate(self.key_list) if len(v) >= 2 ]
        self.larger_than_1_vocab  = [i for i, v in enumerate(self.key_list) if len(v) >= 1 ]
        assert len(self.larger_than_2_vocab) > 0, 'the datastore is too sparse to conduct a good baseline'

        ## add up the cluster centroid into the cluster
        for i, keys in enumerate(self.key_list):
            if len(keys) > 0:
                self.key_list[i].append(torch.tensor(keys).float().mean(dim=0).half().numpy())
                self.val_list[i].append(self.val_list[i][0])

    def __getitem__(self, idx):
        idx = idx % len(self.larger_than_2_vocab)
        pivot_sample = self.key_list[idx][-1]
        positive_sample = sample(self.key_list[idx][:-1], 1)[0]
 
        while True:
            idx_neg = sample(self.larger_than_1_vocab, 1)[0]
            if idx_neg != idx:
                break

        idx_neg_subidx = sample(range(len(self.key_list[idx_neg])), 1)[0]
        negative_sample = self.key_list[idx_neg][idx_neg_subidx]
        negative_vocab = self.val_list[idx_neg][idx_neg_subidx]

        batch_dict = {
            'negative_samples': torch.tensor(negative_sample),
            'negative_ids': negative_vocab,
            'positive_samples': torch.tensor(positive_sample),
            'positive_ids': idx,
            'pivot_samples': torch.tensor(pivot_sample),
            'pivot_ids': idx,
        }
        return batch_dict

    def __len__(self):
        return len(self.larger_than_2_vocab)


@register_model("transformer_knn_projection")
class TransformerProjection(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

        self.args = args
        self.supports_align_args = True

    def load_state_dict(self, state_dict, strict=True, args=None):
        """
        we rewrite the load state dict here for only load part of trained model
        """
        if self.decoder.knn_lambda_type == 'trainable' or self.decoder.knn_temperature_type == 'trainable' \
                or self.decoder.use_knn_datastore:

            self.upgrade_state_dict(state_dict)
            from fairseq.checkpoint_utils import prune_state_dict
            
            new_state_dict = prune_state_dict(state_dict, args)

            model_dict = self.state_dict()

            print('----------------drop part of model-----------------')
            if args.only_train_knn_parameter:
                remove_keys = []
                
                for k, v in new_state_dict.items():
                    if k not in model_dict or "retrieve_result_to_k_and_lambda" in k:
                        ## the retrieve_result_to_k_and_lambda* layers are not used unless in the final knn-mt tuning phrase.
                        ## so the retrieve_result_to_k_and_lambda* modules must be reset!
                        remove_keys.append(k)
                    elif args.convex_svd and "knn_projection" in k:
                        remove_keys.append(k)
                    print(k)
                
                for k in remove_keys:
                    new_state_dict.pop(k)

            print('----------------knn load part of model-----------------')
            print('----------------loaded part of model------------------')
            for k, v in new_state_dict.items():
                print(k)

            model_dict.update(new_state_dict)

            print('-----------------added knn layer-----------------')
            for k, v in model_dict.items():
                if k not in new_state_dict.keys():
                    print(k)
            device = v.device

            print('-----------------added svd layer-----------------')
            if args.svd:
                has_svd = False
                for k, v in model_dict.items():
                    if 'svd_layer' in k:
                        has_svd = True
                        break
                if not has_svd:
                    ## initialization from convex svd results
                    print('--- initial the svd layer --')
                    for k, v in state_dict.items():
                        if "output_projection" in k:
                            break
                    assert k is not None
                    out_pro_weight = state_dict[k] # vocab, hidden_size
                    self.decoder.useful_vocab, inter_dim, u, vh = self.decoder.build_convex_svd_layer(out_pro_weight, args)

                    self.decoder.svd_layer = self.decoder.build_svd_layer(
                        input_dim=out_pro_weight.shape[1],
                        output_dim=self.decoder.useful_vocab,
                        inter_dim=inter_dim,
                        device=device
                    )

                    svd_layer_weights = [vh, u]
                    svd_layer_name = []
                    for name, key in self.named_parameters():
                        if "svd_layer" in name:
                            svd_layer_name.append(name)
                    
                    for k, v in zip(svd_layer_name, svd_layer_weights):
                        model_dict[k] = Parameter(torch.tensor(v.astype(np.float32))).to(device)

                    print('svd initial done.')

            return super().load_state_dict(model_dict)
        else:
            return super().load_state_dict(state_dict, strict, args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on
        # args for use knn and Training knn parameters

        parser.add_argument("--load-knn-datastore", default=False, action='store_true')
        parser.add_argument("--dstore-filename", default=None, type=str)
        parser.add_argument("--use-knn-datastore", default=False, action='store_true')

        parser.add_argument("--dstore-fp16", action='store_true', default=False, help="if save only fp16")
        parser.add_argument("--dstore-size", metavar="N", default=1, type=int, help="datastore size")
        parser.add_argument("--k", default=8, type=int)
        parser.add_argument("--probe", default=32, type=int)

        parser.add_argument("--faiss-metric-type", default=None, type=str)
        parser.add_argument("--knn-sim-func", default=None, type=str)

        parser.add_argument("--use-gpu-to-search", default=False, action="store_true")
        parser.add_argument("--no-load-keys", default=False, action="store_true")
        parser.add_argument("--move-dstore-to-mem", default=False, action="store_true")
        parser.add_argument("--only-use-max-idx", default=False, action="store_true")

        parser.add_argument("--knn-lambda-type", default="fix", type=str)
        parser.add_argument("--knn-lambda-value", default=0.5, type=float)
        parser.add_argument("--knn-lambda-net-hid-size", default=0, type=int)

        parser.add_argument("--label-count-as-feature", default=False, action="store_true")
        parser.add_argument("--relative-label-count", default=False, action="store_true")
        parser.add_argument("--knn-net-dropout-rate", default=0.5, type=float)

        # parser.add_argument("--knn-lambda-net-input-label-count", default=)
        parser.add_argument("--knn-temperature-type", default="fix", type=str)
        parser.add_argument("--knn-temperature-value", default=10, type=float)
        parser.add_argument("--knn-temperature-net-hid-size", default=0, type=int)

        # we add 4 arguments for trainable k network
        parser.add_argument("--knn-k-type", default="fix", type=str)
        parser.add_argument("--max-k", default=None, type=int)
        parser.add_argument("--knn-k-net-hid-size", default=0, type=int)
        parser.add_argument("--knn-k-net-dropout-rate", default=0, type=float)

        # we add 3 arguments for trainable k_with_lambda network
        parser.add_argument("--k-lambda-net-hid-size", type=int, default=0)
        parser.add_argument("--k-lambda-net-dropout-rate", type=float, default=0.0)
        parser.add_argument("--gumbel-softmax-temperature", type=float, default=1)
        parser.add_argument("--knn-use-last-layer-weight", default=False, action="store_true")

        parser.add_argument("--avg-k", default=False, action='store_true')

        parser.add_argument("--only-train-knn-parameter", default=False, action='store_true')
        parser.add_argument("--fcg", default=False, action='store_true')
        parser.add_argument('--decoder-knn-compact-dim', type=int, metavar='N', help='decoder knn compact dimension')
        parser.add_argument("--save-knn-compate-feature", default=False, action='store_true')
        # parser.add_argument("--not-train-knn-compact-projection", default=False, action='store_true')
        # parser.add_argument("--svd", default=False, action='store_true')
        # parser.add_argument("--train-svd", default=False, action='store_true')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
 
        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        # we disable the parameter updated other than knn parameter
        if args.not_train_knn_compact_projection and not args.train_svd:
            
            print('---------trainable params----------') 
            if args.only_train_knn_parameter:
                for name, param in encoder.named_parameters():
                    param.requires_grad = False
                
                for name, param in decoder.named_parameters():
                    param.requires_grad = False
                
                for name, param in decoder.named_parameters():
                    
                    if "knn_distance_to_lambda" in name and decoder.knn_lambda_type == "trainable":
                        param.requires_grad = True
                        print('layer %s with param num %d' % (name, param.numel()))

                    if "knn_distance_to_k" in name and decoder.knn_k_type == "trainable":
                        param.requires_grad = True
                        print('layer %s with param num %d' % (name, param.numel()))
                        
                    if "retrieve_result_to_k_and_lambda" in name \
                            and decoder.knn_lambda_type == "trainable" \
                            and decoder.knn_k_type == "trainable":
                        param.requires_grad = True
                        print('layer %s with param num %d' % (name, param.numel()))

        ## NOTE in svd training  only the svd layer should be update.
        ## NOTE the output_projection layer should not be updated
        if args.train_svd:
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            for name, param in decoder.named_parameters():
                param.requires_grad = False
            for name, param in decoder.named_parameters():
                if "svd_layer" in name:
                    param.requires_grad = True
        
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)

        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            features_and_prob: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            train_knn_projection: bool = True,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        if train_knn_projection and self.decoder.train_knn_compact_projection:
            decoder_out = self.decoder(prev_output_tokens, train_knn_projection=train_knn_projection)
            return decoder_out

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            features_and_prob=features_and_prob,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            train_knn_projection=train_knn_projection,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        # trainable (distances to lambda and temperature) knn datastore
        self.fp16 = args.fp16        
        self.use_knn_datastore = args.use_knn_datastore
        self.knn_lambda_type = args.knn_lambda_type
        self.knn_temperature_type = args.knn_temperature_type
        self.knn_k_type = args.knn_k_type
        self.label_count_as_feature = args.label_count_as_feature
        self.relative_label_count = args.relative_label_count
        self.avg_k = args.avg_k
        self.knn_use_last_layer_weight = args.knn_use_last_layer_weight
        self.fcg = args.fcg
        self.save_knn_compate_feature = args.save_knn_compate_feature
        self.dictionary = dictionary

        ## adaptive knn part
        if self.knn_lambda_type == "trainable" and self.knn_k_type == "trainable":
            # TODO 
            last_layer_dim = self.output_projection.weight.shape[1]
            last_layer_pro_dim = 1
            self.retrieve_result_to_k_and_lambda_last_layer = nn.Sequential(
                nn.Linear(last_layer_dim, 4),
                nn.Tanh(),
                nn.Dropout(p=args.k_lambda_net_dropout_rate),
                nn.Linear(4, last_layer_pro_dim),
            )
            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda_last_layer[0].weight, gain=0.01)
            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda_last_layer[0].weight, gain=0.1)


            # TODO another network to predict k and lambda at the same time without gumbel softmax
            input_dim = args.max_k if not self.label_count_as_feature else args.max_k * 2
            if self.knn_use_last_layer_weight:
                input_dim += last_layer_pro_dim * args.max_k
            self.retrieve_result_to_k_and_lambda = nn.Sequential(
                nn.Linear(input_dim, args.k_lambda_net_hid_size),
                nn.Tanh(),
                nn.Dropout(p=args.k_lambda_net_dropout_rate),
                nn.Linear(args.k_lambda_net_hid_size, 2 + int(math.log(args.max_k, 2))),
                # nn.Softmax(dim=-1),  # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
                # nn.Sigmoid(),  # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
            )
            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, : args.k], gain=0.01)
            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, args.k:], gain=0.1)

        else:
            if self.knn_lambda_type == 'trainable':
                # TODO, we may update the label count feature here
                self.knn_distances_to_lambda = nn.Sequential(
                    nn.Linear(args.k if not self.label_count_as_feature else args.k * 2, args.knn_lambda_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=args.knn_net_dropout_rate),
                    nn.Linear(args.knn_lambda_net_hid_size, 1),
                    nn.Sigmoid())

                if self.label_count_as_feature:
                    # nn.init.normal_(self.knn_distances_to_lambda[0].weight[:, :args.k], mean=0, std=0.01)
                    # nn.init.normal_(self.knn_distances_to_lambda[0].weight[:, args.k:], mean=0, std=0.1)

                    nn.init.xavier_normal_(self.knn_distances_to_lambda[0].weight[:, : args.k], gain=0.01)
                    nn.init.xavier_normal_(self.knn_distances_to_lambda[0].weight[:, args.k:], gain=0.1)
                    nn.init.xavier_normal_(self.knn_distances_to_lambda[-2].weight)

                else:
                    nn.init.normal_(self.knn_distances_to_lambda[0].weight, mean=0, std=0.01)

            if self.knn_temperature_type == 'trainable':
                # TODO, consider a reasonable function
                self.knn_distance_to_temperature = nn.Sequential(
                    nn.Linear(args.k + 2, args.knn_temperature_net_hid_size),
                    nn.Tanh(),
                    nn.Linear(args.knn_temperature_net_hid_size, 1),
                    nn.Sigmoid())
                # the weight shape is [net hid size, k + 1)
                nn.init.normal_(self.knn_distance_to_temperature[0].weight[:, :-1], mean=0, std=0.01)
                nn.init.normal_(self.knn_distance_to_temperature[0].weight[:, -1:], mean=0, std=0.1)

            # TODO we split the network here for different function, but may combine them in the future
            if self.knn_k_type == "trainable":
                self.knn_distance_to_k = nn.Sequential(
                    nn.Linear(args.max_k * 2 if self.label_count_as_feature else args.max_k,
                              args.knn_k_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=args.knn_k_net_dropout_rate),
                    # nn.Linear(args.knn_k_net_hid_size, args.max_k),
                    nn.Linear(args.knn_k_net_hid_size, args.max_k),
                    nn.Softmax(dim=-1))

                # nn.init.xavier_uniform_(self.knn_distance_to_k[0].weight, gain=0.01)
                # nn.init.xavier_uniform_(self.knn_distance_to_k[-2].weight, gain=0.01)
                # # TODO this maybe change or remove from here
                if self.label_count_as_feature:
                    nn.init.normal_(self.knn_distance_to_k[0].weight[:, :args.max_k], mean=0, std=0.01)
                    nn.init.normal_(self.knn_distance_to_k[0].weight[:, args.max_k:], mean=0, std=0.1)
                else:
                    nn.init.normal_(self.knn_distance_to_k[0].weight, mean=0, std=0.01)

        ''' XXX knn compact projection part '''
        ## NOTE we train and test our compact projection layer in a quick way by fetching 
        ## input from the original 1024-d knn datastore,
        ## rather than from  MT feed-forward output from source-/target- text.
        ## if we want to train the knn_compact_projection layer
        args.train_knn_compact_projection = not args.not_train_knn_compact_projection
        self.train_knn_compact_projection = args.train_knn_compact_projection        
        self.test_knn_compact_projection = args.test_knn_compact_projection
        args.dimension = args.decoder_knn_compact_dim if not self.train_knn_compact_projection \
            and not self.test_knn_compact_projection else args.decoder_output_dim 

        ## build compact/non-compact knn datastore
        self.knn_datastore = None
        if args.load_knn_datastore and not args.create_knn_compact_projection:
            self.knn_datastore = KNN_Dstore(args, len(dictionary))

        no_bussiness_with_ori_1024_datastore = not self.train_knn_compact_projection \
            and not self.test_knn_compact_projection

        if args.create_knn_compact_projection or no_bussiness_with_ori_1024_datastore:
            self.dataiterator = None
        else:
            if PYTORCH_DATAAPI: 
                dataiterator = DataLoader(
                    dataset=TripletDatastoreSamplingDataset(args, dictionary, use_cluster=True, knn_datastore=self.knn_datastore),
                    batch_size=1024,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True)
                def get_batch():
                    while True:
                        for i, data in enumerate(dataiterator):
                            yield data
                self.dataiterator = get_batch()
            else:
                self.dataiterator = self.knn_training_iterator(args, dictionary, use_cluster=True)

        ## define the knn compact projection layer
        input_dim = self.output_projection.weight.shape[1]
        output_dim = args.decoder_knn_compact_dim

        self.cknn_compact_layer = self.build_knn_compact_layer(input_dim, output_dim, args.k_lambda_net_dropout_rate)
        if self.train_knn_compact_projection:
            self.cknn_word_pred_layer = nn.Linear(output_dim, len(dictionary), bias=False)
            nn.init.normal_(self.cknn_word_pred_layer.weight, mean=0, std=output_dim ** -0.5)

        ## define the svd projection layer
        self.svd = args.svd
        self.train_svd = False if not self.svd else args.train_svd
        self.convex_svd = False if not self.svd else args.convex_svd
        self.useful_vocab = 10000
        if self.svd:
            if self.convex_svd:
                if self.train_svd:
                    self.svd_layer = None
                else:
                    self.useful_vocab, inter_dim, u, v = self.build_convex_svd_layer(self.output_projection.weight, args)
                    self.svd_layer = self.build_svd_layer(
                        input_dim=args.decoder_output_dim,
                        output_dim=self.useful_vocab,
                        inter_dim=inter_dim,
                    )
            else:
                self.svd_layer = self.build_svd_layer(
                    self.output_projection.weight.shape[1],
                    self.useful_vocab,
                    dropout=args.k_lambda_net_dropout_rate
                )
        self.network_lambda = []
            
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def build_svd_layer(self, input_dim, output_dim, inter_dim=64, dropout=0., device=None):
        if self.train_svd:
            svd_layer = nn.Sequential(
                nn.Linear(input_dim, inter_dim, bias=False),
                nn.Dropout(p=dropout),
                nn.Linear(inter_dim, output_dim, bias=False),
            )
            nn.init.xavier_normal_(svd_layer[0].weight, gain=0.01)
            nn.init.xavier_normal_(svd_layer[-1].weight, gain=0.1) 
        else:
            svd_layer = nn.Sequential(
                nn.Linear(input_dim, inter_dim, bias=False),
            )
            nn.init.xavier_normal_(svd_layer[-1].weight, gain=0.1)
        if device is not None:
            svd_layer = svd_layer.to(device)
        return svd_layer

    def build_convex_svd_layer(self, X, args):
        '''
        Use SVD matrix to initialize the SVD layer.
        The function returns SVD results, not direct initializes the model layer.
        '''
        useful_vocab = self.useful_vocab # or all_total_vocab: X.shape[0]
        u, s, vh = np.linalg.svd(X[:useful_vocab].detach().numpy(), full_matrices=False)
        '''
        NOTE: The shape of u,s,vh:
        when full_matrices=True:
            u -> vocab * vocab
            s -> vocab * hidden_size
            vh -> hidden_size * hidden_size            
        when full_matrices=False:
            u -> vocab * hidden_size
            s -> hidden_size
            vh -> hidden_size * hidden_size
        '''

        # s_sum = s.sum()
        # s_sum_prefix = 0.
        # energy = 0.8 # 0.8: 10000*1024->681 42000*1024->690
        # energy_idx = 0
        # for i in range(s.shape[0]):
        #     s_sum_prefix += s[i]
        #     if s_sum_prefix / s_sum >= energy:
        #         energy_idx = i
        #         break
        # energy_idx += 1
        # print('--- seleceted energy dimension = %d' % energy_idx)

        energy_idx = args.decoder_knn_compact_dim
        s = s[:energy_idx]
        smat = np.zeros((energy_idx, energy_idx), dtype=complex)
        smat[:energy_idx, :energy_idx] = np.diag(s)
        vh = vh[:energy_idx]
        vh = np.dot(smat, vh[:energy_idx]) # energy_idx, hidden_size
        u = u[:, :energy_idx] # vocab, energy_idx

        return useful_vocab, energy_idx, u, vh

    def build_knn_compact_layer(self, input_dim, output_dim, dropout=0.):
        dropout = 0.
        cknn_compact_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(), 
            nn.Dropout(p=dropout),
            nn.Linear(input_dim // 4, output_dim),
        )
        nn.init.xavier_normal_(cknn_compact_layer[0].weight, gain=0.01)
        nn.init.xavier_normal_(cknn_compact_layer[-1].weight, gain=0.1)
        return cknn_compact_layer

    def knn_augment_generator(self, args, verbose = False):
        '''
        TODO: deprecated generator
        '''
        print('Loading to memory...')

        dtype_1024 = np.float16 if args.dstore_fp16 else np.float32
        keys_from_memmap = np.memmap(args.dstore_filename + '/keys.npy', dtype=dtype_1024, mode='r', shape=(args.dstore_size, 1024))
        self.keys = np.zeros((args.dstore_size, 1024), dtype=dtype_1024)
        self.keys = keys_from_memmap[:].astype(dtype_1024)

        self.vals_from_memmap = np.memmap(args.dstore_filename + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
        self.vals = np.zeros((args.dstore_size, 1), dtype=np.int)
        self.vals = self.vals_from_memmap[:].astype(np.int)

        self.vals = torch.from_numpy(self.vals)
        if torch.cuda.is_available():
            print('put vals to gpu')
            self.vals = self.vals.cuda()

        dictionary = self.dictionary
        vocab_freq = [0 for _ in range(len(dictionary))]
        key_list = [[] for _ in range(len(dictionary))]
        val_list = [[] for _ in range(len(dictionary))]

        ## frequence collection
        # for i in tqdm(range(args.dstore_size // 1000)): # for debugging
        for i in tqdm(range(args.dstore_size)):
            val = self.vals[i]
            vocab_freq[val] += 1
            key_list[val].append(self.keys[i])
            val_list[val].append(val)

        ## TODO: iterator for training knn projection layer
        non_empty_key_num = len([k for k in key_list if len(k) != 0])
        from copy import deepcopy
        
        def knn_augment(batch_size: int):
            up_bond = 8092 * 5
            low_bond = 4

            dstore_size = [low_bond if freq < low_bond and freq != 0 else freq for freq in vocab_freq ] 
            dstore_size = [up_bond if freq > up_bond else freq for freq in dstore_size]
            dstore_size = sum(dstore_size)

            print('Datsore size = %d' % (dstore_size))
            print('Saving fp32')
            print('Total useful vocabularies are %d' % non_empty_key_num)

            print(args.dstore_filename + '/knn_transfered_nce_augment/keys.npy')
            print('the datastore is {}'.format(args.dstore_filename + '/knn_transfered_nce_augment/keys.npy'))

            dtype = np.float16 # if self.fp16 else np.float32
            torch_dtype = torch.float32 # torch.float16 if self.fp16 else torch.float32

            dstore_keys = np.memmap(args.dstore_filename + '/knn_transfered_nce_augment/keys.npy', dtype=np.float16, mode='w+',
                                    shape=(dstore_size, args.decoder_knn_compact_dim))
            dstore_vals = np.memmap(args.dstore_filename + '/knn_transfered_nce_augment/vals.npy', dtype=np.int, mode='w+',
                                    shape=(dstore_size, 1))

            var = 0.05
            dstore_idx = 0
            target, tmp_sample = [], []
            for i, (keys, vals) in enumerate(tqdm(zip(key_list, val_list))):
                if len(keys) == 0:
                    continue

                true_len = len(keys)

                if true_len > up_bond:
                    tmp_sample.extend(sample(keys, up_bond))
                    true_len = up_bond
                elif true_len >= low_bond:
                    tmp_sample.extend(keys)
                else:
                    extend_k = deepcopy(keys)
                    for _ in range(low_bond // true_len):
                        tmp_keys = deepcopy(keys)
                        for i in range(true_len):
                            tmp_keys[i] += np.random.rand(keys[0].shape[0]) * var
                        extend_k.extend(tmp_keys)
                    tmp_sample.extend(extend_k[:low_bond])
                    true_len = low_bond
                target.extend([vals[0] for _ in range(true_len)])

                batch_len = len(tmp_sample)
                if batch_len < batch_size:
                    continue

                tmp_sample = torch.tensor(tmp_sample).cuda().to(torch_dtype)
                
                knn_projected_fea = self.cknn_compact_layer(tmp_sample)
                dstore_keys[dstore_idx:batch_len + dstore_idx] = knn_projected_fea.detach().cpu().numpy().astype(dtype)
                dstore_vals[dstore_idx:batch_len + dstore_idx] = torch.tensor(target)[:, None].cpu().numpy().astype(np.int)
                dstore_idx += batch_len

                target, tmp_sample = [], []

            batch_len = len(tmp_sample)
            if batch_len > 0:
                tmp_sample = torch.tensor(tmp_sample).cuda().to(torch_dtype)
                
                knn_projected_fea = self.cknn_compact_layer(tmp_sample)
                dstore_keys[dstore_idx:batch_len + dstore_idx] = knn_projected_fea.detach().cpu().numpy().astype(dtype)
                dstore_vals[dstore_idx:batch_len + dstore_idx] = torch.tensor(target)[:, None].cpu().numpy().astype(np.int)
                dstore_idx += batch_len

            print('the datastore size is %s' % (dstore_idx))
            print('Datastore augmentation done!')

        knn_augment(batch_size=2048)
        
    def knn_training_iterator(
            self,
            args,
            dictionary = None,
            use_cluster : bool = False,
            cluster_type : str = 'dbscan',
            sample_rate :float = 1.0,
            verbose :bool = False
        ):

        assert sample_rate <= 1.0

        db_keys, db_vals = self.knn_datastore.keys, self.knn_datastore.vals

        if sample_rate != 1.0:
            random_sample = int(sample_rate * db_vals.shape[0])
            db_keys = db_keys[:random_sample]
            db_vals = db_vals[:random_sample]
            args.dstore_size = random_sample
        
        vocab_freq = [0 for _ in range(len(dictionary))]
        key_list = [[] for _ in range(len(dictionary))]

        ## frequence collection
        # for i in tqdm(range(args.dstore_size // 500)): # for debugging or saving time
        for i in tqdm(range(args.dstore_size)):
            val = db_vals[i]
            vocab_freq[val] += 1
            key_list[val].append(db_keys[i])
        del db_keys
        del db_vals
        del self.knn_datastore

        ## clustering
        if use_cluster:
            
            ## inner clustering refine
            cluster_algorithm_list = ['spectrum', 'dbscan']
            assert cluster_type in cluster_algorithm_list, 'the cluster algorithm should be in the list: ' + ' '.join(cluster_algorithm_list)
            
            if cluster_type == 'spectrum':
                from sklearn.cluster import SpectralClustering
                sc = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', n_init=3, n_neighbors=4)
            elif cluster_type == 'dbscan':
                from sklearn.cluster import DBSCAN
                sc = DBSCAN(eps=10, min_samples=min_samples)

            print('start clustering ...')
            new_key_list = []
            new_val_list = []
            base_number = min_samples

            # Limited by memory, 100000 koran/it/medical (<=10M) 20000 for law/subtitles (>=19M). 
            sample_bound = 100000
            for vocab_id, keys in tqdm(enumerate(key_list)):
                if len(keys) == 0:
                    continue

                if vocab_id % 2000 == 0:
                    print('clustering %d' % vocab_id)

                '''
                key_list[0] is a list of all-zero keys, because the key and value mmap space were initialized as all zero tensors.
                key_list[1~3] are not all-zero keys, of which the vocabs are '<pad> </s> <unk>'
                '''
                if vocab_id < 4 and vocab_id != 2:
                    continue

                if len(keys) <= base_number:
                    new_key_list.append(keys)
                    new_val_list.append([vocab_id for _ in range(len(keys))])
                    continue

                ## to decrease the computation
                if len(keys) > sample_bound:
                    keys = sample(keys, sample_bound)

                sc.n_clusters = int(math.log(len(keys)+base_number, base_number))
                sc.n_neighbors = min(len(keys), min_samples)

                keys = np.array(keys)

                clustering = sc.fit(keys)
                labels = clustering.labels_

                tmp_key = [[] for _ in range(labels.max()+1)]
                for n in range(labels.shape[0]):
                    if labels[n] == -1:
                        continue
                    tmp_key[labels[n]].append(keys[n])
                    # print(labels[j], end=' ')
                tmp_key = [key for key in tmp_key if len(key) != 0]
                new_key_list.extend(tmp_key)

                tmp_val = [[vocab_id for _ in range(len(key))] for key in tmp_key]
                new_val_list.extend(tmp_val)
                assert len(tmp_key) == len(tmp_val)

            # del key_list
            key_list = new_key_list
            val_list = new_val_list
            '''
            After target-side clustering, tokens of the same vocab may be split
            into different slices of this new_val_list, like:
            [
             [5,5,5], [5,5,5,5,5],
             [6,], [6,6,6,6], [6,6,6], [6,6],
             [7],
             [8,8,8,8], [8,8],
              ...
            ]
            '''

            print('we get %d clusters' % len(key_list))

            # # post-processing
            # for i in range(len(key_list)):
            #     if len(key_list[i]) == 0:
            #         continue
            #     key_list[i] = np.array(key_list[i])

            print('cluster done. Get %d nodes' % sum([len(keys) for keys in key_list]))
        del sc

        ## statistics collection of vocab frequency
        larger_than_2_vocab  = [i for i, v in enumerate(key_list) if len(v) >= 2 ]
        larger_than_1_vocab  = [i for i, v in enumerate(key_list) if len(v) >= 1 ]
        assert len(larger_than_2_vocab) > 0, 'the datastore is too sparse to conduct a good baseline'

        ## add up the cluster centroid into the cluster
        for i, keys in enumerate(key_list):
            if len(keys) > 0:
                key_list[i].append(torch.tensor(keys).float().mean(dim=0).half().numpy())
                val_list[i].append(val_list[i][0])

        ## XXX iterator for training knn projection layer
        def knn_projection_triplet_iterator(batch_size):
            assert len(larger_than_2_vocab) >= batch_size, 'the number of vocabs %d whose frequence >= 2 is less than batch_size %d' % (len(larger_than_2_vocab), batch_size)
            while True:
                pivot_samples, positive_samples, negative_samples = [], [], []
                pivot_ids, positive_ids, negative_ids = [], [], []

                # idx = randint(0, len(dictionary)-1)
                # if vocab_freq[idx] < 2: continue
                idxs = sample(larger_than_2_vocab, batch_size)
                for idx in idxs:
                    # postive_pair = sample(key_list[idx], 2)
                    # ## get pivot
                    # pivot_sample = postive_pair[0]
                    # ## get positive
                    # positive_sample = postive_pair[1]
                    # ## get negative 
                    # # idx_neg = randint(0, len(dictionary)-1)
                    # idx_neg = sample(larger_than_1_vocab, 1)[0]
                    # negative_sample = sample(key_list[idx_neg], 1)[0]

                    pivot_sample = key_list[idx][-1]
                    positive_sample = sample(key_list[idx][:-1], 1)[0]

                    while True:
                        idx_neg = sample(larger_than_1_vocab, 1)[0]
                        if idx_neg != idx:
                            break

                    idx_neg_subidx = sample(range(len(key_list[idx_neg])), 1)[0]
                    negative_sample = key_list[idx_neg][idx_neg_subidx]
                    negative_vocab = val_list[idx_neg][idx_neg_subidx]

                    negative_ids.append(negative_vocab)
                    negative_samples.append(negative_sample)
                    positive_samples.append(positive_sample)
                    pivot_samples.append(pivot_sample)


                negative_ids = torch.tensor(negative_ids).long()
                pivot_ids = torch.tensor([val_list[idx][-1] for idx in idxs]).long()
                positive_ids = pivot_ids

                batch_dict = {
                    'negative_samples': torch.tensor(negative_samples),
                    'negative_ids': negative_ids,
                    'positive_samples': torch.tensor(positive_samples),
                    'positive_ids': positive_ids,
                    'pivot_samples': torch.tensor(pivot_samples),
                    'pivot_ids': pivot_ids,
                }
                yield batch_dict

        return knn_projection_triplet_iterator(batch_size=1024)

    def forward_svd_training(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            # use_knn_store: bool = False
        ):

        select_output_projection = self.output_projection.weight[:self.useful_vocab]


        select_u = self.svd_layer[0].weight.T.contiguous()
        select_v = self.svd_layer[-1].weight.T.contiguous()

        svd = torch.matmul(select_u, select_v)

        # print(svd[-1, :15] - select_output_projection[-1, :15])

        bias = svd.T.contiguous() - select_output_projection
        
        l1_bias = bias * ((bias > 0.).float() - 0.5) # self.useful_vocab, hidden_size
        l1_bias = l1_bias.sum(0) / bias.shape[-1]
        dis_loss = l1_bias.mean(-1)

        # l2_bias = bias * bias
        # l2_bias = l2_bias.sum(0) / bias.shape[-1]
        # dis_loss = l2_bias.mean(-1)
        
        unused_tensor = torch.tensor([0.]).to(dis_loss.device)
        return dis_loss, unused_tensor, dis_loss * unused_tensor, \
            unused_tensor, unused_tensor, \
            unused_tensor


    def forward_knn_training(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            # use_knn_store: bool = False
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        ## to make compact feature space, we do compact projection.
        device = prev_output_tokens.device
        data = next(self.dataiterator)
        negative_samples = data['negative_samples']
        pivot_samples = data['pivot_samples']
        positive_samples = data['positive_samples']

        negative_ids = data['negative_ids']
        pivot_ids = data['pivot_ids']
        positive_ids = data['positive_ids']

        batch_size = negative_samples.shape[0]

        stack_ids = torch.cat([pivot_ids, positive_ids, negative_ids], dim=0).to(device)
        stack_data = torch.cat([pivot_samples, positive_samples, negative_samples], dim=0).to(device).float()

        projected_data = self.cknn_compact_layer(stack_data)
        pivot, positive, negative = projected_data[:batch_size], projected_data[batch_size:2*batch_size], projected_data[2*batch_size:3*batch_size]

        ## I. distance ranking loss        
        # split into multi head to avoid overfitting
        '''
        XXX multi head might be bad in this case
        '''
        # head = 4
        # pivot_multi_head = pivot.view(batch_size * head, -1)
        # positive_multi_head = positive.view(batch_size * head, -1)
        # negative_multi_head = negative.view(batch_size * head, -1)
        # pos_dis = nn.MSELoss(reduction='mean')(pivot_multi_head, positive_multi_head)
        # neg_dis = nn.MSELoss(reduction='mean')(pivot_multi_head, negative_multi_head)

        # non-split version
        pos_dis = nn.MSELoss(reduce=False)(pivot, positive).sum(-1)

        margin = 10.
        def hingle_loss(margin, pivot, negative):
            neg_dis = nn.MSELoss(reduce=False)(pivot, negative).sum(-1)
            neg_dis = (neg_dis < margin).float() * neg_dis + (neg_dis >= margin).float() * margin
            return neg_dis
        neg_dis = hingle_loss(margin, pivot, negative)

        # compute weighted l2 ranking loss
        soft_pos = 1. # pos_dis / (pos_dis + neg_dis)
        soft_neg = 1. # 1. - soft_pos
        soft_pos_dis = soft_pos * pos_dis
        soft_reverse_neg_dis = soft_neg * (margin / (neg_dis + 1e-3))
        dis_ranking_loss = soft_pos_dis + soft_reverse_neg_dis
        dis_ranking_loss = dis_ranking_loss.mean()

        '''
        NOTE: l2 ranking loss supplymentary loss for distinguishing vocab of extremely low frequency
        '''
        negative_drop_first = negative[1:].view((batch_size - 1), -1)
        negative_drop_last =  negative[:-1].view((batch_size - 1), -1)
        extra_neg_dis = hingle_loss(margin, negative_drop_first, negative_drop_last)
        dis_ranking_loss += (margin / (extra_neg_dis + 1e-3)).mean()
        dis_ranking_loss *= 0. # eliminate dis_ranking loss

        ## II. nce loss inside positive/negative samples
        ## 1. distance-based nce
        # nce_feature = positive[:, None, :] - pivot[None, :, :] # bsz, bsz, h
        # nce_feature = negative[:, None, :] - pivot[None, :, :] # bsz, bsz, h
        # nce_distance = (nce_feature * nce_feature).sum(-1)

        ## 2. dot-based nce 
        nce_distance_pos = - (positive[:, None, :] * pivot[None, :, :]).sum(-1) # bsz, bsz
        nce_distance = nce_distance_pos
        '''
        NOTE the simplest nce is to optimize among positive pairs in a batch, but sampling of positive
        pairs ignore tokens of low frequence Which make the optimization only done for high-frequence vocab.
        To address this, we optimize positive pairs nce loss along with negative pairs
        '''
        nce_distance_neg = - (negative[:, None, :] * pivot[None, :, :]).sum(-1) # bsz, bsz
        nce_distance = torch.cat((nce_distance_pos, nce_distance_neg), axis=1)

        nce_lprobs = torch.nn.functional.log_softmax(-nce_distance, dim=-1) # the larger, the worse
        nce_target = torch.arange(end=batch_size).to(device)
        nce_loss, _ = label_smoothed_nll_loss(nce_lprobs, nce_target, reduce=True)
        nce_loss = nce_loss / float(batch_size)

        ## III. prediction loss
        # word_logits = self.cknn_word_pred_layer(projected_data)
        # word_probs = torch.nn.functional.log_softmax(word_logits, dim=-1) # the larger, the worse
        # word_pred_loss, _ = label_smoothed_nll_loss(word_probs, stack_ids, reduce=True)
        # word_pred_loss = word_pred_loss / float(batch_size)
        # nce_loss += word_pred_loss

        return dis_ranking_loss, nce_lprobs, nce_loss, pivot, positive, negative

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            features_and_prob: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            train_knn_projection: bool = True,
            # use_knn_store: bool = False
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        
        ## train the knn compact projection layer
        if train_knn_projection and self.train_knn_compact_projection:
            knn_align_train_res = self.forward_knn_training(prev_output_tokens)
            return knn_align_train_res
        
        if train_knn_projection and self.train_svd:
            svd_train_res = self.forward_svd_training(prev_output_tokens)
            return svd_train_res
        
        ## extract mt features or train adaptive-mt
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer, 
            alignment_heads=alignment_heads,
        )

        bsz = x.shape[0]
        seq_len = x.shape[1]
        device = x.device
        
        if self.svd:
            ## to get svd refiner features
            x_knn = self.svd_layer[0](x)
        else:
            ## to get knn refiner features
            x_knn = self.cknn_compact_layer(x)

        if self.use_knn_datastore:
            # last_hidden = x # [B, S, 1024]
            last_hidden = x_knn        
        
        if self.save_knn_compate_feature:
            if features_and_prob:
                return x_knn, nn.Softmax(dim=-1)(self.output_layer(x))
            return x_knn, None
        
        if not features_only:
            if features_and_prob:
                return x, nn.Softmax(dim=-1)(self.output_layer(x))
            x = self.output_layer(x)
        # return x, None 
        if self.use_knn_datastore:
            # we should return the prob of knn search
            knn_search_result = self.knn_datastore.retrieve(last_hidden)

            # knn_probs = knn_search_result['prob']
            knn_dists = knn_search_result['distance'] # [batch, seq len, k]  # we need sort
            knn_index = knn_search_result['knn_index']
            tgt_index = knn_search_result['tgt_index']

            if self.label_count_as_feature:
                # get the segment label count here
                label_counts = self.knn_datastore.get_label_count_segment(tgt_index, relative=self.relative_label_count)
                network_inputs = torch.cat((
                    knn_dists.detach(),
                    label_counts.detach().float()),
                    dim=-1) # [B, S, 2*K]

                if self.knn_use_last_layer_weight:
                    selected_last_layer_weights = last_hidden[:,:, None, :] - functional.embedding(tgt_index, self.output_projection.weight) # lookup and sub
                    model_derived_fea = self.retrieve_result_to_k_and_lambda_last_layer(
                        selected_last_layer_weights.detach()
                    ).reshape(network_inputs.shape[0], network_inputs.shape[1], -1)
                    network_inputs = torch.cat((network_inputs, model_derived_fea), dim=-1)
            else:
                network_inputs = knn_dists.detach()

            if self.fp16:
                network_inputs = network_inputs.half()

            if self.knn_temperature_type == 'trainable':
                knn_temperature = None
            else:
                knn_temperature = self.knn_datastore.get_temperature()
            
            if self.knn_lambda_type == "trainable" and self.knn_k_type == 'trainable':

                net_outputs = self.retrieve_result_to_k_and_lambda(network_inputs) # [B, S, K]

                k_prob = nn.Softmax(dim=-1)(net_outputs)

                # we add this here only to test the effect of avg prob
                if self.avg_k:
                    k_prob = torch.zeros_like(k_prob).fill_(1. / k_prob.size(-1))

                # knn_lambda = 1. - k_prob[:, :, 0: 1]
                # k_soft_prob = k_prob[:, :, 1:]

                nmt_base_prob = 0.# 0.5
                knn_lambda = 1. - k_prob[:, :, 0: 1]

                k_soft_prob = k_prob[:, :, 1:]
                knn_lambda = knn_lambda * (1. - nmt_base_prob)
                k_soft_prob = k_soft_prob * (1. - nmt_base_prob)

                # ## if use sigmoid instead of softmax
                # if not features_only and self.fcg:
                #     row_id_broadcaster = torch.arange(bsz * seq_len).unsqueeze(-1).expand(bsz * seq_len, self.knn_datastore.k).contiguous()
                #     row_id_broadcaster = row_id_broadcaster.view(bsz * seq_len * self.knn_datastore.k).to(device)
                #     col_id = tgt_index.view(-1)
                #     temp_x = x.view(bsz * seq_len, -1).softmax(-1).cpu()[row_id_broadcaster, col_id].view(bsz, seq_len, -1)
                #     k_soft_prob = (1. - temp_x) * k_soft_prob
                
                decode_result = self.knn_datastore.calculate_select_knn_prob(knn_index, tgt_index, knn_dists,
                                                                            last_hidden,
                                                                            knn_temperature,
                                                                            k_soft_prob,
                                                                            is_test=not self.retrieve_result_to_k_and_lambda.training)
                # return x, extra

            else:
                if self.knn_lambda_type == 'trainable':
                    # self.knn_distances_to_lambda[2].p = 1.0
                    knn_lambda = self.knn_distances_to_lambda(network_inputs)

                else:
                    knn_lambda = self.knn_datastore.get_lambda()

                if self.knn_k_type == "trainable":
                    # we should generate k mask
                    k_prob = self.knn_distance_to_k(network_inputs)

                    if self.knn_distance_to_k.training:
                        # print(k_prob[0])
                        k_log_prob = torch.log(k_prob)
                        k_soft_one_hot = functional.gumbel_softmax(k_log_prob, tau=0.1, hard=False, dim=-1)
                        # print(k_one_hot[0])

                    else:
                        # we get the one hot by argmax
                        _, max_idx = torch.max(k_prob, dim=-1)  # [B, S]
                        k_one_hot = torch.zeros_like(k_prob)
                        k_one_hot.scatter_(-1, max_idx.unsqueeze(-1), 1.)

                        knn_mask = torch.matmul(k_one_hot, self.knn_datastore.mask_for_distance)

                if self.knn_k_type == "trainable" and self.knn_distance_to_k.training:
                    decode_result = self.knn_datastore.calculate_select_knn_prob(knn_index, tgt_index, knn_dists,
                                                                                 last_hidden,
                                                                                 knn_temperature,
                                                                                 k_soft_one_hot)

                elif self.knn_k_type == "trainable":
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists,
                                                                          last_hidden,
                                                                          knn_temperature,
                                                                          knn_mask)

                else:
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists,
                                                                          last_hidden,
                                                                          knn_temperature)

            knn_prob = decode_result['prob']

            # while True:
            #     pass
            
            if self.label_count_as_feature:
                return x, extra, knn_prob, knn_lambda, knn_dists, knn_index, label_counts
            else:
                return x, extra, knn_prob, knn_lambda, knn_dists, knn_index

        else:
            # original situation
            return x, extra

    def predict_on_total_trainset(self):
        '''
        Oracle probing w.r.t. the context feature ranking.
        '''
        dot_logit = None

        datastore_shape = self.keys.shape
        self.cknn_compact_layer = self.cknn_compact_layer.cuda()
        device = self.cknn_compact_layer[0].weight.device
        dsn = datastore_shape[0]

        ## projection phrase
        bsz = 8192
        step = dsn // bsz if dsn % bsz == 0 else dsn // bsz + 1
        projection_keys = []
        
        with torch.no_grad():
            for i in tqdm(range(step)):
                stack_data = torch.from_numpy(self.keys[i*bsz:(i+1)*bsz]).float().to(device)
                
                knn_search_result = self.knn_datastore.retrieve(stack_data[:, None])
                pr = knn_search_result['tgt_index'].cpu()
                gt = self.vals[i*bsz:(i+1)*bsz].cpu()
                compare = (pr == gt).float().sum() / bsz
                input(compare)
                continue
                
                ## knn compact mt
                projected_data = self.cknn_compact_layer(stack_data)

                ## original mt
                # projected_data = stack_data

                projection_keys.append(projected_data.cpu())
                torch.cuda.empty_cache()
            projection_keys = torch.cat(projection_keys, dim=0)

            ## computing inner global similarity phrase, large computation! we split the computation into pieces.
            bsz = 512
            step = dsn // bsz if dsn % bsz == 0 else dsn // bsz + 1
            dot_logit_collection = []
            
            wsz = projection_keys.shape[0]
            
            top_k = []
            w_bsz = 1000
            w_step = wsz // w_bsz if wsz % w_bsz == 0 else wsz // w_bsz + 1


            for i in tqdm(range(step)):
                k = projection_keys[i*bsz:(i+1)*bsz].to(device)
                tmp_dot = []# bsz
                for j in tqdm(range(w_step)):
                    split_projection_keys = projection_keys[j*w_bsz:(j+1)*w_bsz].to(device)

                    ## dot-style distance
                    #dot = k[None] * split_projection_keys[:, None]
                    #dot = dot.sum(-1).cpu()

                    ## l2 distance
                    distance = k[None] - split_projection_keys[:, None]
                    distance = distance * distance
                    dot = distance.sum(-1).cpu() # w_bsz, bsz
                    
                    tmp_dot.append(dot) 
                    
                    split_projection_keys = None
                    torch.cuda.empty_cache()
                tmp_dot = torch.cat(tmp_dot, dim=0)
                # tmp_dot_min = torch.min(tmp_dot, dim=0)[1]
                tmp_dot_min_value, tmp_dot_min = torch.topk(tmp_dot, k=2, dim=0, largest=False, sorted=False, out=None)
                

                pr = self.vals[tmp_dot_min[-1]].cpu()
                gt = self.vals[np.arange(i*bsz, (i+1)*bsz)].cpu() 
                compare = (pr == gt).float().sum() / bsz
                input(compare)
                top_k.append(tmp_dot_min)

            top_k = torch.cat(top_k, dim=0)

            print(top_k.shape)
            print(top_k[:100])
            input()

        return acc

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output. we modify this method to return prob with
        knn result
        """

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out
        logits = net_output[0]

        # wo combine the knn prob and network prob here
        if self.use_knn_datastore:
            # x, extra, knn_probs, knn_lambda

            knn_probs = net_output[2]  # [batch, seq len, vocab size]
            knn_lambda = net_output[3]  # [batch, seq len, 1]
            network_probs = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)  # [batch, seq len, vocab size]
            # return network_probs
            if self.knn_lambda_type == "fix":
                probs = network_probs * (1 - knn_lambda) + knn_probs * knn_lambda
            else:
                # debugging
                # knn_probs = (knn_probs > network_probs).float() * knn_probs
                # mt_argmax = torch.argmax(network_probs, -1, keepdim=False)
                # to_keep_knn = (mt_argmax == 3).float()
                # probs = network_probs * (1 - knn_lambda * to_keep_knn[:, :, None]) + knn_probs * to_keep_knn[:, :, None]

                probs = network_probs * (1 - knn_lambda) + knn_probs

                # debugging
                # self.network_lambda.append((1 - knn_lambda).cpu().numpy())
                # probs = network_probs * 1. + 0. * knn_probs # for debugging
                # print(knn_lambda)
                # probs = probs + 1e-6 # to avoid inf
                # with open('network_lambda.txt', 'a+') as f:
                #     f.write(1 - knn_lambda.cpu().numpy())
                # print(np.mean(network_lambda))
                
            if log_probs:
                return torch.log(probs)
            else:
                return probs

        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_knn_compact_dim = getattr(args, "decoder_knn_compact_dim", args.decoder_knn_compact_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)



def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_knn_compact_dim = getattr(args, "decoder_knn_compact_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer_knn_projection", "transformer_knn_de_en_transfered_by_distance")
def transformer_knn_de_en_transfered_by_distance(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    transformer_vaswani_wmt_en_de_big(args)
