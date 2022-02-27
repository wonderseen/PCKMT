#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os 
import sys 
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(args, override_args=None):
    
    utils.import_user_module(args)
    assert (
            args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None
    
    # Load ensemble
    # the task is build based on the checkpoint
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    logger.info(model_args)
    
    # Build criterion, we do not need this, so remove it, by
    # criterion = task.build_criterion(model_args)
    # criterion.eval()
    if args.save_plain_text:
        batch_src_tokens = []
        batch_target = []

    # --- check save data store
    import numpy as np
    if args.dstore_fp16:
        print('Saving fp16')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='w+',
                                shape=(args.dstore_size, args.decoder_knn_compact_dim))
    else:
        print('Saving fp32')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='w+',
                                shape=(args.dstore_size, args.decoder_knn_compact_dim))
    dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int64, mode='w+',
                            shape=(args.dstore_size, 1))
    destore_tgt_lengths = np.memmap(args.dstore_mmap + '/tgt_lens.npy', dtype=np.int64, mode='w+',
                            shape=(args.dstore_size, 1))
    destore_src_lengths = np.memmap(args.dstore_mmap + '/src_lens.npy', dtype=np.int64, mode='w+',
                            shape=(args.dstore_size, 1))
    dstore_vals_4_gram = np.memmap(args.dstore_mmap + '/vals_4_gram.npy', dtype=np.int64, mode='w+',
                            shape=(args.dstore_size, 4))  
    dstore_vals_4_gram_prob = np.memmap(args.dstore_mmap + '/vals_4_gram_probs.npy', dtype=np.float32, mode='w+',
                            shape=(args.dstore_size, 4))                         
    dstore_vals_entropy = np.memmap(args.dstore_mmap + '/vals_entropy.npy', dtype=np.float32, mode='w+',
                            shape=(args.dstore_size, 1))
    dstore_idx = 0
    # --- end

    data_idx = 1
    for subset in args.valid_subset.split(","):
        try: 
            task.args.required_seq_len_multiple = 1
            task.args.load_alignments = False
            task.load_dataset(subset, combine=False, epoch=data_idx)
            data_idx = data_idx + 1
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(False)
        
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        log_outputs = []
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample

                features, probs = task.forward_and_get_hidden_state_step_and_prob(sample, model)  # [B, T, H]
                target = sample['target']  # [B, T]

                batch_size = target.size(0)
                seq_len = target.size(1)

                def get_4_gram(w):
                    batch_size = w.size(0)
                    w = w[:, :, None]
                    w_pad_1 = torch.cat((torch.zeros((batch_size, 1, 1)).to(torch.long).cuda(), w[:, :-1]), 1)
                    w_pad_2 = torch.cat((torch.zeros((batch_size, 2, 1)).to(torch.long).cuda(), w[:, :-2]), 1)
                    w_pad_3 = torch.cat((torch.zeros((batch_size, 3, 1)).to(torch.long).cuda(), w[:, :-3]), 1)
                    return torch.cat((w, w_pad_1, w_pad_2, w_pad_3), -1)
                target_4_gram = get_4_gram(target)  # [B, T, 3]

                def get_gt_probs(probs, target):
                    '''
                    probs: [B, T, dictionary]
                    target: [B, T]
                    '''
                    B, T, C = probs.size(0), probs.size(1), probs.size(2)
                    one_hot = torch.arange(0, C).to(target.device)[None, None].repeat(B, T, 1) == target[:, :, None]
                    return (probs * one_hot.float()).sum(-1)
                target_prob = get_gt_probs(probs, target)

                def get_entropy(probs):
                    '''
                    probs: [B, T, dictionary]
                    '''
                    return - (probs * torch.log(probs+1e-7)).sum(-1)
                target_entropy = get_entropy(probs)

                def get_n_gram_prob(target_prob):
                    target_prob = target_prob[:, :, None]
                    w_pad_1 = torch.cat((target_prob[:, :1].repeat(1, 1, 1), target_prob[:, :-1]), 1)
                    w_pad_2 = torch.cat((target_prob[:, :1].repeat(1, 2, 1), target_prob[:, :-2]), 1)
                    w_pad_3 = torch.cat((target_prob[:, :1].repeat(1, 3, 1), target_prob[:, :-3]), 1)
                    return torch.cat((target_prob, w_pad_1,  w_pad_2, w_pad_3), -1)
                n_gram_prob = get_n_gram_prob(target_prob)

                target_length = (target != dataset.tgt_dict.pad()).to(torch.int).sum(-1)
                target_length = target_length[:, None].repeat(1, seq_len).view(batch_size * seq_len)
                source_length = sample['net_input']['src_lengths']
                source_length = source_length[:, None].repeat(1, seq_len).view(batch_size * seq_len)

                pad_idx = task.target_dictionary.pad()
                target_mask = target.ne(pad_idx)  # [B, T]
                # target_mask = target > 3 # [n_count] # filter '<s> <pad> </s> <unk>'

                # remove the pad tokens and related hidden states
                target_mask = target_mask.view(batch_size * seq_len)
                target = target.view(batch_size * seq_len)
                target_4_gram = target_4_gram.view(batch_size * seq_len, 4)
                target_prob = target_prob.contiguous().view(batch_size * seq_len, -1)
                n_gram_prob = n_gram_prob.contiguous().view(batch_size * seq_len, -1)
                target_entropy = target_entropy.contiguous().view(batch_size * seq_len, -1)


                non_pad_index = target_mask.nonzero().squeeze(-1)  # [n_count]
                target = target.index_select(dim=0, index=non_pad_index)  # [n_count]
                target_4_gram = target_4_gram.index_select(dim=0, index=non_pad_index)  # [n_count]
                target_prob = target_prob.index_select(dim=0, index=non_pad_index)  # [n_count]
                n_gram_prob = n_gram_prob.index_select(dim=0, index=non_pad_index)  # [n_count]
                target_entropy = target_entropy.index_select(dim=0, index=non_pad_index)  # [n_count]

                features = features.contiguous().view(batch_size * seq_len, -1)
                features = features.index_select(dim=0, index=non_pad_index)  # [n_count, feature size]

                target_length = target_length.index_select(dim=0, index=non_pad_index)
                source_length = source_length.index_select(dim=0, index=non_pad_index)

                # if save plain text
                if args.save_plain_text:
                    src_tokens = sample['net_input']['src_tokens']  # [B, src len]
                    assert src_tokens.size(0) == batch_size
                    src_len = src_tokens.size(-1)
                    src_tokens = src_tokens.unsqueeze(1).expand(batch_size, seq_len, src_len). \
                        reshape((batch_size * seq_len, src_len))
                    src_tokens = src_tokens.index_select(dim=0, index=non_pad_index)  # [n_count, src_len]
                    batch_src_tokens.append(src_tokens.cpu())
                    batch_target.append(target.cpu().unsqueeze(-1))

                # save to the dstore
                current_batch_count = target.size(0)
                if dstore_idx + current_batch_count > args.dstore_size:
                    reduce_size = args.dstore_size - dstore_idx
                    features = features[:reduce_size]
                    target = target[:reduce_size]
                    target_4_gram = target_4_gram[:reduce_size]
                    target_prob = target_prob[:reduce_size]
                    n_gram_prob = n_gram_prob[:reduce_size]
                    target_entropy = target_entropy[:reduce_size]
                    target_length = target_length[:reduce_size]
                    source_length = source_length[:reduce_size]
                    if args.save_plain_text:
                        src_tokens = src_tokens[:reduce_size, :]
                else:
                    reduce_size = current_batch_count

                if args.dstore_fp16:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(np.float16)
                else:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(np.float32)
                dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.unsqueeze(-1).cpu().numpy().astype(
                    np.int)
                destore_tgt_lengths[dstore_idx:reduce_size + dstore_idx] = target_length.unsqueeze(-1).cpu().numpy().astype(
                    np.int)
                destore_src_lengths[dstore_idx:reduce_size + dstore_idx] = source_length.unsqueeze(-1).cpu().numpy().astype(
                    np.int)
                dstore_vals_4_gram[dstore_idx:reduce_size + dstore_idx] = target_4_gram.detach().cpu().numpy().astype(
                    np.int)
                dstore_vals_4_gram_prob[dstore_idx:reduce_size + dstore_idx] = n_gram_prob.detach().cpu().numpy().astype(
                    np.float32)
                dstore_vals_entropy[dstore_idx:reduce_size + dstore_idx] = target_entropy.detach().cpu().numpy().astype(
                    np.float32)

                dstore_idx += reduce_size

                if dstore_idx >= args.dstore_size:
                    print('much more than dstore size break')
                    break
            # -------- end, by

        print('filtered dstore size = %d' % (dstore_idx+1))


    if args.save_plain_text:

        # Set dictionaries
        try:
            src_dict = getattr(task, "source_dictionary", None)
        except NotImplementedError:
            src_dict = None
        tgt_dict = task.target_dictionary

        plain_text = []

        for src_tokens, target in tqdm(zip(batch_src_tokens, batch_target)):

            src_strs = src_dict.string(src_tokens, return_list=True, extra_symbols_to_ignore=[src_dict.pad()])  # [[str]]
            trg_tokens = tgt_dict.string(target, return_list=True)  # [[token]]
            cur_trg_str = ""

            for src_str, trg_token in zip(src_strs, trg_tokens):
                if len(trg_token) == 0:
                    _trg_token = "<eos>"
                else:
                    _trg_token = trg_token
                cur_trg_str = cur_trg_str + ' {}'.format(_trg_token)
                plain_text.append("src: {} trg: {}".format(src_str, cur_trg_str))
                if len(trg_token) == 0:
                    cur_trg_str = ""

        with open(args.dstore_mmap + '/text.txt', 'w') as f:
            for line in plain_text:
                f.write(f"{line}\n")


def cli_main():
    parser = options.get_save_datastore_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)
    distributed_utils.call_main(args, main, override_args=override_args)

 
if __name__ == "__main__":
    cli_main()
