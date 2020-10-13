import argparse, os, time

import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt

from data import *


def tp_one_trail(dataset, model_type, topic_size, sample_size,
             max_iter=1000, min_iter=None, checkpoint=None, stop_increase=10, metric='ll_word'):
    assert model_type in ['lda', 'ctm'], f'invalid `model_type`: {model_type}...'
    assert metric in ['ll', 'perplexity'], f'invalid `metric`: {metric}...'
    if model_type == 'lda':
        model = tp.LDAModel(k=topic_size)
    if model_type == 'ctm':
        model = tp.CTModel(k=topic_size)

    sample_size = min(sample_size, len(dataset))
    for i in range(sample_size):
        doc, label = dataset[i]
        model.add_doc(doc)

    if min_iter is None:
        min_iter = max_iter // 10
    if checkpoint is None:
        checkpoint = max_iter // 10

    for i in range(min_iter):
        model.train(i)

    pre_metric = - np.infty
    stop_increase_cnt = 0.
    for i in range(0, max_iter, checkpoint):
        model.train(i)
        if metric == 'll':
            cur_metric = model.ll_per_word
        if metric == 'perplexity':
            cur_metric = model.perplexity
        if cur_metric >= pre_metric:
            pre_metric = cur_metric
        else:
            stop_increase_cnt += 1

        if stop_increase_cnt >= stop_increase:
            break

    return model


def boxplot_results(results_data, results_title, figwidth=5, figheight=4):
    result = np.array(results_data).T

    plt.figure(figsize=(figwidth, figheight))
    _ = plt.boxplot(result)
    plt.xticks(range(1, len(results_data) + 1), results_title)
    plt.savefig(plt_path)


def eval_model(model, dataset, metric='ll'):
    docs = [model.make_doc(x) for (x, _) in dataset]
    lda_x, llk = model.infer(docs)
    if metric == 'll':
        return llk
    if metric == 'perplexity':
        # TODO: how to compute perplexity?
        return llk


def run_trails(args, choice_set):
    start = time.time()
    trainset = IMDBDataset('train')
    print(f'Finish: prepare train set (size: {len(trainset)}) in {(time.time() - start):.3f} seconds.')

    start = time.time()
    validset = IMDBDataset('valid')
    print(f'Finish: prepare valid set (size: {len(validset)}) in {(time.time() - start):.3f} seconds.')

    results_data = []
    results_title = []

    for (n, k) in choice_set:
        if n > len(trainset):
            continue
        start = time.time()
        cur_result = []

        print(f'start trail: {n}&{k}.')
        for _ in range(args.rep_times):
            trained_model = tp_one_trail(trainset, args.model, k, n,
                                         args.max_iter, args.min_iter, args.checkpoint,
                                         args.stop_increase, args.metric)
            cur_result.append(eval_model(trained_model, validset, args.metric))

        cur_result = np.array(cur_result)
        results_title.append(f'{n}&{k}')
        try:
            results_data.append(cur_result.mean(0))
        except:
            print(cur_result.shape)
            exit(1)
        print(f'Finish: [{n}&{k}] trails in {(time.time() - start):.3f} seconds.')

    boxplot_results(results_data, results_title)


if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    result_path = os.path.join(cur_path, 'result')

    parser = argparse.ArgumentParser(description="IWAE experiment")
    parser.add_argument("--model", type=str, choices=['lda', 'ctm'], default='lda')
    parser.add_argument("--task", type=str, choices=['n', 'k', 'nk'], default='k')
    parser.add_argument("--rep_times", type=int, default=3)
    # train
    parser.add_argument("--max_iter", type=int, default=1_000)
    parser.add_argument("--min_iter", type=int, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--stop_increase", type=int, default=10)
    parser.add_argument("--metric", type=str, default='ll')
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=100)

    args = parser.parse_args()

    if args.task == 'n':
        ns = [5_000, 10_000, 20_000, 40_000]
#         ns = [500, 1000, 2000]
        choice_set = [(n, args.k) for n in ns]
    if args.task == 'k':
        ks = [10, 50, 100, 300]
        choice_set = [(args.n, k) for k in ks]
    if args.task == 'nk':
        choice_set = [(5_000, 10), (10_000, 50), (20_000, 100), (40_000, 300)]


    plt_path = os.path.join(result_path, args.model, args.task)
    start = time.time()
    print('Start trails.')
    run_trails(args, choice_set)
    print(f"""Finish trails in {(time.time() - start):.3f} seconds, 
            See result at {plt_path}...""")