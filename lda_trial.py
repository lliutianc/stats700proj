import argparse, os, time, pickle

import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt

from data import *
from util import *


def tp_one_trial(dataset, model_type, topic_size, sample_size, min_cf=3, rm_top=5, burn_in=500,
             max_iter=1000, min_iter=None, checkpoint=None, stop_increase=1, metric='ll'):
    assert model_type in ['lda', 'ctm',"slda"], f'invalid `model_type`: {model_type}...'
    assert metric in ['ll', 'pp'], f'invalid `metric`: {metric}...'
    if model_type == 'lda':
        model = tp.LDAModel(k=topic_size, tw=tp.TermWeight.ONE, min_cf=min_cf, rm_top=rm_top)
    if model_type == 'ctm':
        model = tp.CTModel(k=topic_size, tw=tp.TermWeight.ONE, min_cf=min_cf, rm_top=rm_top)
    if model_type == "slda":
        model = tp.SLDAModel(k=topic_size,vars="b", tw=tp.TermWeight.ONE, min_cf=min_cf, rm_top=rm_top)
    model.burn_in = burn_in
    sample_size = min(sample_size, len(dataset))
    for i in range(sample_size):
        doc, label = dataset[i]
        if model_type == "slda":
            model.add_doc(doc,[float(label),])
        else:
            model.add_doc(doc)

    if min_iter is None:
        min_iter = max_iter // 10
    if checkpoint is None:
        checkpoint = max_iter // 50

    model.train(min_iter)

    pre_metric = - np.infty
    stop_increase_cnt = 0.
    cur_metric = 0.
    for i in range(1, max_iter+1):
        model.train(1)
        # Metric is always larger, better
        if metric == 'll':
            cur_metric += model.ll_per_word
        if metric == 'pp':
            cur_metric += - model.perplexity  # smaller perplexity is better.

        if i % checkpoint == 0:
            cur_metric /= checkpoint
            print(f'Current loss: {cur_metric:.5f}')
            if cur_metric >= pre_metric:
                pre_metric = cur_metric
            else:
                stop_increase_cnt += 1
            cur_metric = 0.

        if stop_increase_cnt >= stop_increase:
            break

    final_metric = model.perplexity if metric == 'pp' else model.ll_per_word

    print(f'Trial iterations: {i + min_iter}.')
    return model, final_metric


def boxplot_results(results_train_metric,
                    results_train, results_valid, results_title,
                    args, figwidth=5, figheight=4):
    # results_train_metric = np.array(results_train_metric)
    results_train = np.array(results_train).T
    results_train_metric = np.median(results_train, axis=0)

    results_valid = np.array(results_valid).T

    plt.cla()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # ax_r = plt.twinx()
    _ = ax.boxplot(results_valid)
    _ = ax.plot(np.arange(1, len(results_train_metric) + 1), results_train_metric, label=f'trainset: {args.metric}')
    ax.set_xticks(range(1, len(results_train_metric) + 1))
    ax.set_xticklabels(results_title)
    ax.set_xlabel(f'N & K')
    ax.set_ylabel(f'{args.metric}')
    ax.legend()
    
    plt.tight_layout()
    plt_path = result_path + f'/{args.model}-{args.task}-{args.metric}.jpg'
    plt.savefig(plt_path)


def eval_model(model, dataset, metric='ll'):
    assert metric in ['ll', 'pp'], f'invalid `metric`: {metric}...'

    docs = [model.make_doc(x) for (x, _) in dataset]
    lda_x, llk = model.infer(docs)
    n = np.array([len(x) for (x, _) in dataset])
    llk_per_word = llk / n
    if metric == 'll':
        return llk_per_word
    if metric == 'pp':
        return np.exp(-llk_per_word)


def run_trials(args, choice_set):
    start = time.time()
    trainset = IMDBDataset('train', data_limit=20_000)
    # trainset.shuffle()
    print(f'Finish: prepare train set (size: {len(trainset)}) in {(time.time() - start):.3f} seconds.')

    start = time.time()
    validset = IMDBDataset('valid')
    print(f'Finish: prepare valid set (size: {len(validset)}) in {(time.time() - start):.3f} seconds.')

    results_train = []
    results_valid = []
    results_title = []
    results_train_metric = []
    for (n, k) in choice_set:
        if n > len(trainset):
            continue
        start = time.time()
        cur_train, cur_valid = [], []
        cur_train_metric = 0.
        print(f'start trial: {n}&{k}.')
        for r in range(args.rep_times):
            trained_model, final_metric = tp_one_trial(trainset, args.model, k, n, 
                                                       args.min_cf, args.rm_top, args.burn_in,
                                                       args.max_iter, args.min_iter, args.checkpoint,
                                                       args.stop_increase, args.metric)

            cur_train.append(eval_model(trained_model, trainset, args.metric))
            cur_valid.append(eval_model(trained_model, validset, args.metric))

            trained_model.save(os.path.join(result_path, f'{args.model}#{n}#{k}#{r}.bin'), full=False)
            cur_train_metric += final_metric

        results_train_metric.append(cur_train_metric / args.rep_times)
        cur_train = np.array(cur_train)
        cur_valid = np.array(cur_valid)
        results_title.append(f'{n}&{k}')
        try:
            results_train.append(cur_train.mean(0))
            results_valid.append(cur_valid.mean(0))
        except:
            print(cur_valid.shape)
            exit(1)
        print(f'Finish: [{n}&{k}] trials in {(time.time() - start):.3f} seconds.')

    trial_result = {'results_train_metric': results_train_metric,
                    'results_train': results_train,
                    'results_valid': results_valid,
                    'results_title': results_title}
    with open(os.path.join(result_path, f'{args.model}-{args.task}.pkl'), 'wb') as file:
        pickle.dump(trial_result, file)

    boxplot_results(results_train_metric, results_train, results_valid, results_title, args)
    # try:
    #     # boxplot_results(results_train_metric, results_data, results_title, args)
    # except:
    #     print(f'Fail to plot: {n}#{k}.')


if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    result_path = os.path.join(cur_path, 'result')

    make_dirs(result_path)

    parser = argparse.ArgumentParser(description="Stat700proj experiment")
    parser.add_argument("--model", type=str, choices=['lda', 'ctm', "slda"], default='lda')
    parser.add_argument("--task", type=str, choices=['n', 'k', 'nk'], default='k')
    parser.add_argument("--rep_times", type=int, default=3)
    # train
    parser.add_argument("--max_iter", type=int, default=2_000)
    parser.add_argument("--min_iter", type=int, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--stop_increase", type=int, default=5)
    parser.add_argument("--metric", type=str, default='pp')
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=20)
    # Other
    parser.add_argument("--rm_top", type=int, default=5)
    parser.add_argument("--min_cf", type=int, default=3)
    parser.add_argument("--burn_in", type=int, default=500)

    args = parser.parse_args()

    ns = [1_000, 3_000, 5_000, 10_000, 15_000, 20_000]
    ks = [1, 2, 3, 5, 10, 50]

    if args.task == 'n':
        choice_set = [(n, args.k) for n in ns]
    if args.task == 'k':
        choice_set = [(args.n, k) for k in ks]
    if args.task == 'nk':
        choice_set = list(zip(ns, ks))

    start = time.time()
    print(f'Start trials: {args.model}-{args.task}')
    run_trials(args, choice_set)
    print(f"Finish trials in {(time.time() - start):.3f} seconds.")
