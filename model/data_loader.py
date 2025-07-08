# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, dataset='S_VideoXum', split_num=0):
        """ Custom Dataset class wrapper for loading the frame features, text features and ground truth importance scores.
        :param str mode: The mode of the model, train, val or test.
        :param str dataset: The name of the dataset: ['S_VideoXum' | 'S_NewsVSum']
        """
        self.mode = mode
        if dataset == 'S_VideoXum':
            self.filename = './dataset/script_videoxum.h5'
            self.dataset_split = './dataset/script_videoxum_split.json'

        elif dataset == 'S_NewsVSum':
            self.filename = './dataset/S_NewsVSum.h5'
            self.dataset_split = './dataset/S_NewsVSum_split.json'
        else:
            raise ValueError("Error: no valid dataset. Must be: ['S_VideoXum' | 'S_NewsVSum']")
        hdf = h5py.File(self.filename, 'r')

        with open(self.dataset_split, 'r') as f:
            keys = json.load(f)
        if dataset == 'S_NewsVSum':
            keys = keys[split_num]

        self.list_video_features, self.list_text_features, self.list_gtscores, self.list_video_name = [], [], [], []

        for video_name in hdf.keys():
            if video_name in keys[self.mode]:
                video_features = torch.Tensor(np.array(hdf[video_name + '/video_embeddings'][()]))
                text_features = torch.Tensor(np.array(hdf[video_name + '/text_embeddings'][()]))
                gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscores']))

                self.list_video_name.append(video_name)
                self.list_video_features.append(video_features)
                self.list_text_features.append(text_features)
                self.list_gtscores.append(gtscore)

        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of the dataset. """
        return len(self.list_video_name)

    def __getitem__(self, index):
        """ Function to be called for the index operator of the dataset.
        :param int index: The above-mentioned id of the data.
        """
        video_features = self.list_video_features[index]
        text_features = self.list_text_features[index]
        gtscore = self.list_gtscores[index]
        return video_features, text_features, gtscore


def get_loader(mode, dataset='S_VideoXum', split_num=0):
    """ Loads the dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str dataset: The name of the dataset: ['S_VideoXum' | 'S_NewsVSum']
    :return: The dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, dataset=dataset)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode)


if __name__ == '__main__':
    pass