#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:37:27 2020

@author: fabian
"""


import numpy as np
import random
from sklearn.utils import shuffle
import torch.utils.data as data
import torch        
from imblearn import over_sampling, under_sampling, combine


class CustomDataLoader(data.Dataset):
    def __init__(self, data, labels, augment=False, nclasses=27,
                 balance=False, split='train'):
        self.nclasses = nclasses
        self.data = data
        self.labels = labels
        self.augment = augment
        self.balance = balance
        self.split = split
        if balance:
            # At data[:, 0] is a list containing all pressure frames
            # At data[:, 1] is a list containing the corresponding IMU data
            self.original_data = data
            #self.original_data[:, 0] = self.original_data[:, 0].reshape((len(labels),32*32))
            self.original_labels = labels
            self.balance_data()
        self.collate_data()

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        pressure_imu = []
        pressure_imu.append(self.collated_data[idx][0])
        pressure_imu[0] = pressure_imu[0].reshape((32, 32))
        pressure_imu[0] = np.expand_dims(pressure_imu[0], axis=0)
        pressure_imu[0] = torch.from_numpy(pressure_imu[0])
        pressure_imu.append(torch.from_numpy(self.collated_data[idx][1]))
        if self.augment:
            noise = torch.randn_like(pressure_imu[0]) * 0.015#0.015
            pressure_imu[0] += noise
        object_id = torch.LongTensor([int(self.collated_labels[idx])])
        return pressure_imu, object_id


    def collate_data(self):
        """
        Function to collate the training or test data into blocks that are
        correctly shaped

        Returns
        -------
        None.

        """
        
        self.collated_data = self.data
        self.collated_labels = self.labels
        # shuffle
        self.collated_data,\
            self.collated_labels = shuffle(self.collated_data,
                                           self.collated_labels)
        return

        
    def balance_data(self):
        # Randomize for every refresh call
        seed = random.randint(0,1000)
        if self.split == 'train':
            # # Randomize for every refresh call
            # neighbors = random.randint(2,4)
            # clusters = random.randint(16,24)
    
            # oversampler = over_sampling.KMeansSMOTE(random_state=seed,
            #                                         kmeans_estimator=clusters,
            #                                         k_neighbors=neighbors,
            #                                         sampling_strategy='not majority')
            # # oversampler = over_sampling.SVMSMOTE(random_state=seed, out_step=step,
            # #                                      k_neighbors=5, m_neighbors=10)
            
            # # Undersample majority class to the second largest class
            # # class 0 is always the biggest class (empty hand)
            # nmax = 0
            # for i in range(1, self.nclasses):
            #     mask = self.original_labels == i
            #     n = np.count_nonzero(mask)
            #     if n > nmax:
            #         nmax = n
            # strat = {0: nmax}
            # undersampler = under_sampling.RandomUnderSampler(random_state=seed,
            #                                                   sampling_strategy=strat)
            # sampler = combine.SMOTETomek(random_state=seed, kmeans_estimator=clusters,
            #                               k_neighbors=neighbors)
            
            undersampler = under_sampling.RandomUnderSampler(random_state=seed,
                                                             sampling_strategy='not minority')
            
            resampled_data,\
                resampled_labels = undersampler.fit_resample(self.original_data,
                                                             self.original_labels)
            # try:
            #     resampled_data,\
            #         resampled_labels = oversampler.fit_resample(resampled_data,
            #                                                     resampled_labels)
            # except:
            #     neighbors = 2
            #     clusters = 24
            #     oversampler = over_sampling.KMeansSMOTE(random_state=seed,
            #                                             kmeans_estimator=clusters,
            #                                             k_neighbors=neighbors,
            #                                             sampling_strategy='not majority')
            #     resampled_data,\
            #         resampled_labels = oversampler.fit_resample(resampled_data,
            #                                                     resampled_labels)
                
        elif self.split == 'test':
            resampled_data = self.original_data
            resampled_labels = self.original_labels
                
        self.data = resampled_data
        self.labels = resampled_labels
 
    
    def refresh(self):
        print('Refreshing dataset...')
        if self.balance:
            self.balance_data()
        self.collate_data()
