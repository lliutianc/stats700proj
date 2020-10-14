import argparse, os, time, pickle

import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt

from data import *
from util import *


def tp_one_trial(dataset, model_type, topic_size, sample_size,
             max_iter=1000, min_iter=None, checkpoint=None, stop_increase=10, metric='ll_word'):
    assert model_type in ['lda', 'ctm',"slda"], f'invalid `model_type`: {model_type}...'
    assert metric in ['ll', 'pp'], f'invalid `metric`: {metric}...'
    if model_type == 'lda':
        model = tp.LDAModel(k=topic_size)
    if model_type == 'ctm':
        model = tp.CTModel(k=topic_size)
    if model_type == "slda":
        model = tp.SLDAModel(k=topic_size,vars="b")

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
        checkpoint = max_iter // 10

    for i in range(min_iter):
        model.train(i)

    pre_metric = - np.infty
    stop_increase_cnt = 0.
    for i in range(0, max_iter, checkpoint):
        model.train(checkpoint)
        if metric == 'll':
            cur_metric = model.ll_per_word
        if metric == 'pp':
            cur_metric = - model.perplexity  # smaller perplexity is better.
        if cur_metric >= pre_metric:
            pre_metric = cur_metric
        else:
            stop_increase_cnt += 1

        if stop_increase_cnt >= stop_increase:
            break

    final_metric = model.perplexity if metric == 'pp' else model.ll_per_word
    return model, final_metric


def boxplot_results(results_train_metric, results_data, results_title, args, figwidth=5, figheight=4):
    results_train_metric = np.array(results_train_metric)
    result = np.array(results_data).T

    plt.figure(figsize=(8, 6))
    _ = plt.boxplot(result, legend='vald_value')
    _ = plt.plot(range(1, len(results_data) + 1), results_train_metric, label='train_value')
    plt.xticks(range(1, len(results_data) + 1), results_title)
    plt.xlabel(f'N & K')
    plt.ylabel(f'{args.metric}')
    plt.legend()
    
    plt.tight_layout()
    plt_path = result_path + f'/{args.model}-{args.task}.jpg'
    plt.savefig(plt_path)


def eval_model(model, dataset, metric='ll'):
    assert metric in ['ll', 'pp'], f'invalid `metric`: {metric}...'

    docs = [model.make_doc(x) for (x, _) in dataset]
    lda_x, llk = model.infer(docs)
    if metric == 'll':
        return llk
    if metric == 'pp':
        n = np.array([len(x) for (x, _) in dataset])
        pp = llk / n
        return np.exp(-pp)


def run_trials(args, choice_set):
    start = time.time()
    trainset = IMDBDataset('train')
    print(f'Finish: prepare train set (size: {len(trainset)}) in {(time.time() - start):.3f} seconds.')

    start = time.time()
    validset = IMDBDataset('valid')
    print(f'Finish: prepare valid set (size: {len(validset)}) in {(time.time() - start):.3f} seconds.')

    results_data = []
    results_title = []
    results_train_metric = []
    for (n, k) in choice_set:
        if n > len(trainset):
            continue
        start = time.time()
        cur_result = []
        cur_train_metric = 0.
        print(f'start trial: {n}&{k}.')
        for r in range(args.rep_times):
            trained_model, final_metric = tp_one_trial(trainset, args.model, k, n,
                                                       args.max_iter, args.min_iter, args.checkpoint,
                                                       args.stop_increase, args.metric)
            cur_result.append(eval_model(trained_model, validset, args.metric))
            trained_model.save(os.path.join(result_path, f'{args.model}#{n}#{k}#{r}.bin'))
            cur_train_metric += final_metric

        results_train_metric.append(cur_train_metric / args.rep_times)
        cur_result = np.array(cur_result)
        results_title.append(f'{n}&{k}')
        try:
            results_data.append(cur_result.mean(0))
        except:
            print(cur_result.shape)
            exit(1)
        print(f'Finish: [{n}&{k}] trials in {(time.time() - start):.3f} seconds.')

    trial_result = {'results_train_metric': results_train_metric,
                    'results_data': results_data, 
                    'results_title': results_title}
    with open(os.path.join(result_path, f'{args.model}-{args.task}.pkl'), 'wb') as file:
        pickle.dump(trial_result, file)
    try:
        boxplot_results(results_train_metric, results_data, results_title, args)
    except:
        print(f'Fail to plot: {plt_path}.')


if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    result_path = os.path.join(cur_path, 'result')
    model_path = os.path.join(cur_path, 'model')

    make_dirs(result_path)

    parser = argparse.ArgumentParser(description="IWAE experiment")
    parser.add_argument("--model", type=str, choices=['lda', 'ctm', "slda"], default='lda')
    parser.add_argument("--task", type=str, choices=['n', 'k', 'nk'], default='k')
    parser.add_argument("--rep_times", type=int, default=5)
    # train
    parser.add_argument("--max_iter", type=int, default=1_000)
    parser.add_argument("--min_iter", type=int, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--stop_increase", type=int, default=10)
    parser.add_argument("--metric", type=str, default='ll')
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=50)

    args = parser.parse_args()

    ns = [3_000, 5_000, 10_000, 20_000, 40_000]
    ks = [3, 5, 10, 30, 100]

    if args.task == 'n':
        choice_set = [(n, args.k) for n in ns]
    if args.task == 'k':
        choice_set = [(args.n, k) for k in ks]
    if args.task == 'nk':
        choice_set = list(zip(ns, ks))
        # choice_set = [(5_000, 3), (10_000, 10), (20_000, 30), (40_000, 100)]

    start = time.time()
    print('Start trials.')
    run_trials(args, choice_set)
    print(f"""Finish trials in {(time.time() - start):.3f} seconds, 
            See result at [{result_path}].""")