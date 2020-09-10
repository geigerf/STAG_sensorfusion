import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling


def load_data(filename, kfold=3, seed=333, undersample=True, split='random',
              accel_only=False):

    data = sio.loadmat(filename, squeeze_me=True)
    valid_mask = data['valid_flag'] == 1
    # Scale all data to the range [0, 1]
    pressure = data['tactile_data'][valid_mask].astype(np.float32)
    pressure = np.clip((pressure-1510)/(3000-1510), 0.0, 1.0)
    imu = data['IMU_data'][valid_mask].astype(np.float32)
    imu = np.clip((imu+32768)/65535, 0.0, 1.0)
    object_id = data['object_id'][valid_mask]
    
    pressure_imu = []
    for i in range(len(imu)):
        if(accel_only):
            pressure_imu.append([pressure[i], imu[i, 0:3]])
        else:
            pressure_imu.append([pressure[i], imu[i]])
    pressure_imu = np.array(pressure_imu)

    if kfold is not None:
        # Decrease the test size if cross validation is used
        test_size = 0.15
    else:
        kfold = 3
        test_size = 0.33

    if(split == 'random'):
        if(undersample):
            us = under_sampling.RandomUnderSampler(random_state=seed,
                                               sampling_strategy='not minority')
            us_pressure_imu, us_object_id = us.fit_resample(pressure_imu, object_id)
            
            pressure_imu, object_id = us_pressure_imu, us_object_id
    
        # Split the already balanced dataset in a stratified way -> training
        # and test set will still be balanced
        train_data, test_data,\
            train_labels, test_labels = train_test_split(pressure_imu, object_id,
                                                         test_size=test_size,
                                                         random_state=seed,
                                                         shuffle=True,
                                                         stratify=object_id)
        #print(train_data.shape, train_labels.shape)
        # This generates a k fold split in a stratified way.
        # Easy way to do k fold cross validation
        skf = StratifiedKFold(n_splits=kfold, shuffle=True,
                              random_state=seed)
        # train_ind, val_ind = skf.split(train_data, train_labels)
        # skf_gen = skf.split(train_data, train_labels)
        
        return train_data, train_labels, test_data, test_labels, skf
    
    elif(split == 'session'):
        num_sessions = len(np.unique(data['session_id']))
        x = []
        y = []
        valid_sessions = data['session_id'][valid_mask]
        for i in range(num_sessions):
            session_mask = valid_sessions == i
            x.append(pressure_imu[session_mask])
            y.append(object_id[session_mask])
            
        return x, y  