import argparse, os, time, pickle

import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from data import *
from util import *
from lda_trial import *

def main(args):
    trainset = IMDBDataset('train', data_limit=20_000)
    validset = IMDBDataset('valid')

    if args.task == 'k':
        n = 20000
        ks = [2, 5, 10, 20, 50]
        train_accu = [0 for _ in range(len(ks))]
        test_accu = [0 for _ in range(len(ks))]
        times = [0 for _ in range(len(ks))]
        for _ in range(3):
            for idx, k in enumerate(ks):
                print(f'{k}.')
                start = time.time()
                trained_model, final_metric = tp_one_trial(trainset, args.model, k, n, 
                                                          3, 5,  # args.burn_in,
                                                          max_iter=1000, stop_increase=5, metric='ll')
                times[idx] += time.time() - start
                lda_x, lda_y = load_LDA_data_batch(trained_model, trainset)
                model = LinearSVC()
                model.fit(lda_x, lda_y)

                prediction = model.predict(lda_x)
                ground_truth = lda_y
                print(f'Train: {100*accuracy_score(prediction, ground_truth):6.5f}%')
                train_accu[idx] += accuracy_score(prediction, ground_truth)

                lda_x, lda_y = load_LDA_data_batch(trained_model, validset)
                prediction = model.predict(lda_x)
                ground_truth = lda_y
                print(f'Test: {100*accuracy_score(prediction, ground_truth):6.5f}%')
                test_accu[idx] += accuracy_score(prediction, ground_truth)
                print('-' * 100)

    if args.task == 'n':
        k = 20
        ns = [3_000, 5_000, 10_000, 15_000, 20_000]
        train_accu = [0 for _ in range(len(ns))]
        test_accu = [0 for _ in range(len(ns))]
        times = [0 for _ in range(len(ns))]
        for _ in range(3):
            for idx, k in enumerate(ns):
                print(f'{k}.')
                start = time.time()
                trained_model, final_metric = tp_one_trial(trainset, args.model, k, n, 
                                                          3, 5,  # args.burn_in,
                                                          max_iter=1000, stop_increase=5, metric='ll')
                times[idx] += time.time() - start
                lda_x, lda_y = load_LDA_data_batch(trained_model, trainset)
                model = LinearSVC()
                model.fit(lda_x, lda_y)

                prediction = model.predict(lda_x)
                ground_truth = lda_y
                print(f'Train: {100*accuracy_score(prediction, ground_truth):6.5f}%')
                train_accu[idx] += accuracy_score(prediction, ground_truth)

                lda_x, lda_y = load_LDA_data_batch(trained_model, validset)
                prediction = model.predict(lda_x)
                ground_truth = lda_y
                print(f'Test: {100*accuracy_score(prediction, ground_truth):6.5f}%')
                test_accu[idx] += accuracy_score(prediction, ground_truth)
                print('-' * 100)
        
    train_accu = [accu / rep for accu in train_accu]
    test_accu = [accu / rep for accu in test_accu]
    times = [accu / rep for accu in times]
    
    print(train_accu)
    print(test_accu)
    print(times)
    print(task, args.model)


if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    result_path = os.path.join(cur_path, 'result')

    make_dirs(result_path)

    parser = argparse.ArgumentParser(description="Stat700proj experiment")
    parser.add_argument("--model", type=str, choices=['lda', 'ctm', "slda"], default='lda')
    parser.add_argument("--rep_times", type=int, default=3)
    # train
    parser.add_argument("--task", type=str, default='k')
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=20)
    # Other
    parser.add_argument("--rm_top", type=int, default=5)
    parser.add_argument("--min_cf", type=int, default=3)
    # parser.add_argument("--burn_in", type=int, default=500)

    args = parser.parse_args()

    trainset = IMDBDataset('train', data_limit=20_000)
    validset = IMDBDataset('valid')
    
    # n = 20000
    # ks = [2, 5, 10, 20, 50]
    # train_accu = [0 for _ in range(len(ks))]
    # test_accu = [0 for _ in range(len(ks))]
    # times = [0 for _ in range(len(ks))]
    # for _ in range(3):
    #     for idx, k in enumerate(ks):
    #         print(f'{k}.')
    #         start = time.time()
    #         trained_model, final_metric = tp_one_trial(trainset, args.model, k, n, 
    #                                                   3, 5,  # args.burn_in,
    #                                                   max_iter=1000, stop_increase=5, metric='ll')
    #         times[idx] += time.time() - start
    #         lda_x, lda_y = load_LDA_data_batch(trained_model, trainset)
    #         model = LinearSVC()
    #         model.fit(lda_x, lda_y)

    #         prediction = model.predict(lda_x)
    #         ground_truth = lda_y
    #         print(f'Train: {100*accuracy_score(prediction, ground_truth):6.5f}%')
    #         train_accu[idx] += accuracy_score(prediction, ground_truth)

    #         lda_x, lda_y = load_LDA_data_batch(trained_model, validset)
    #         prediction = model.predict(lda_x)
    #         ground_truth = lda_y
    #         print(f'Test: {100*accuracy_score(prediction, ground_truth):6.5f}%')
    #         test_accu[idx] += accuracy_score(prediction, ground_truth)
    #         print('-' * 100)
        
    # train_accu = [accu / rep for accu in train_accu]
    # test_accu = [accu / rep for accu in test_accu]
    # times = [accu / rep for accu in times]
    
    # print(train_accu)
    # print(test_accu)
    # print(times)
    main(args)
