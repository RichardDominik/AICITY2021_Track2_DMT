# encoding: utf-8

import glob
import os
import re
import xml.dom.minidom as XD
import os.path as osp

import numpy as np

from .bases import BaseImageDataset

class AIC_LFTD(BaseImageDataset):
    dataset_dir = 'AIC21/AIC21_Track2_ReID'

    def __init__(self, root='../data', verbose=True, crop_test=False, **kwargs):
        super(AIC_LFTD, self).__init__()
        self.crop_test = crop_test

        feat_paths = [
            'logs/stage2/resnext101a_384/v1/', 'logs/stage2/resnext101a_384/v2/',
            'logs/stage2/101a_384/v1/', 'logs/stage2/101a_384/v2/',
            'logs/stage2/101a_384_recrop/v1/', 'logs/stage2/101a_384_recrop/v2/',
            'logs/stage2/101a_384_spgan/v1/', 'logs/stage2/101a_384_spgan/v2/',
            'logs/stage2/densenet169a_384/v1/', 'logs/stage2/densenet169a_384/v2/',
            'logs/stage2/s101_384/v1/', 'logs/stage2/s101_384/v2/',
            'logs/stage2/se_resnet101a_384/v1/', 'logs/stage2/se_resnet101a_384/v2/',
            'logs/stage2/transreid_256/v1/', 'logs/stage2/transreid_256/v2/',
            # './logs/stage2/swin_transformer/v1/', './logs/stage2/swin_transformer/v2/',
            # './logs/stage2/swin_transformer_spgan/v1/', './logs/stage2/swin_transformer_spgan/v2/',
            # 'logs/stage2/swin_transformer_384/v1/pth29/', 'logs/stage2/swin_transformer_384/v2/pth29/'
        ]

        feat_paths = [os.path.join(root, feat_path) for feat_path in feat_paths]

        train = self._process_npz_train(feat_paths, relabel=True)
        gallery, query = self._process_npz_test(feat_paths)

        if verbose:
            print("=> AIC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _relabel(self, pids):
        pid_container = set()
        for pid in pids:
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def _process_npz_test(self, feat_path_list, relabel=False):
        first_npz_path = os.path.join(feat_path_list[0], 'val_out_arrays.npz')
        data = np.load(first_npz_path)

        num_query = data['num_query']

        query_pids = data['pids'][:num_query]
        data_pids = data['pids'][num_query:]
        query_camids = data['camids'][:num_query]
        data_camids = data['camids'][num_query:]
        query_tids = data['tids'][:num_query]
        data_tids = data['tids'][num_query:]

        query_feats = np.empty([num_query, len(feat_path_list), data['feats'].shape[-1]])
        data_feats = np.empty([len(data_pids), len(feat_path_list), data['feats'].shape[-1]])

        for i, npz_path in enumerate(feat_path_list):
            npz_full_path = os.path.join(feat_path_list[0], 'val_out_arrays.npz')
            data = np.load(npz_full_path)

            query_feats[:, i, :] = data['feats'][:num_query]
            data_feats[:, i, :] = data['feats'][num_query:]

        if relabel:
            pid2label = self._relabel(data_pids)
            dataset = [(f, pid2label[pid], camid, tid) for f, pid, camid, tid in (zip(data_feats, data_pids, data_camids, data_tids))]
        else:
            dataset = [(f, pid, camid, tid) for f, pid, camid, tid in (zip(data_feats, data_pids, data_camids, data_tids))]
        query_dataset = [(f, pid, camid, tid) for f, pid, camid, tid in (zip(query_feats, query_pids, query_camids, query_tids))]

        return dataset, query_dataset

    def _process_npz_train(self, feat_path_list, relabel=False):
        first_npz_path = os.path.join(feat_path_list[0], 'train_out_arrays.npz')
        data = np.load(first_npz_path)

        data_pids = data['pids']
        data_camids = data['camids']
        data_tids = data['tids']

        data_feats = np.empty([len(data_pids), len(feat_path_list), data['feats'].shape[-1]])

        for i, npz_path in enumerate(feat_path_list):
            npz_full_path = os.path.join(feat_path_list[0], 'train_out_arrays.npz')
            data = np.load(npz_full_path)
            data_feats[:, i, :] = data['feats']

        if relabel:
            pid2label = self._relabel(data_pids)
            dataset = [(f, pid2label[pid], camid, tid) for f, pid, camid, tid in (zip(data_feats, data_pids, data_camids, data_tids)) if pid != -1]
        else:
            dataset = [(f, pid, camid, tid) for f, pid, camid, tid in (zip(data_feats, data_pids, data_camids, data_tids)) if pid != -1]

        return dataset


if __name__ == '__main__':
    aic = AIC_LFTD(root='/home/d/dominik3/diplomovka/AICITY2021_Track2_DMT')
