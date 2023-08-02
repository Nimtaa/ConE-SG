
import argparse
import json
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import KGReasoning
from dataloader import TestDataset, TrainDatasetSG, SingledirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing ConE-SG',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--data', type=str, default=None, help="KG data")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int,
                        help="negative entities sampled per query")
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('-cpu', '--cpu_num', default=4, type=int, help="used to speed up torch.dataloader")

    
    return parser.parse_args(args)


def load_data(args):
    '''
    Load queries
    '''
    
    train_queries = torch.load(os.path.join(args.data_path,'train_input.pt'))
    train_answers = torch.load(os.path.join(args.data_path, 'train_output.pt'))
    valid_queries = torch.load(os.path.join(args.data_path, 'val_input.pt'))
    valid_answers = torch.load(os.path.join(args.data_path, 'val_output.pt'))
    test_queries = torch.load(os.path.join(args.data_path, 'test_input.pt'))
    test_answers = torch.load(os.path.join(args.data_path, 'test_output.pt'))
    train_indics = torch.load(os.path.join(args.data_path, 'train_indices.pt'))
    test_indices = torch.load(os.path.join(args.data_path, 'test_indices.pt'))
    val_indices = torch.load(os.path.join(args.data_path, 'val_indices.pt'))
        

    return train_queries, train_answers, valid_queries, valid_answers, test_queries, test_answers, train_indics, val_indices, test_indices



def main(args):
    
    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation



    train_queries, train_answers, valid_queries, valid_answers, test_queries, test_answers, train_indics, val_indices, test_indices = load_data(args)


    # train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
    # TrainDatasetSG(train_queries, nentity, nrelation, args.negative_sample_size, train_answers),
    # batch_size=args.batch_size,
    # shuffle=True,
    # num_workers=args.cpu_num))

    dl = DataLoader(TrainDatasetSG(train_queries, nentity, nrelation, args.negative_sample_size, train_answers))
    print(next(iter(dl)))







if __name__ == '__main__':
    main(parse_args())