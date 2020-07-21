import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling


def load_data(filename, kfold=3, seed=333, undersample=True):
    """
    Function to load the pressure and IMU data with stratified kfold

    Parameters
    ----------
    filename : string
        Path to the data file.
    kFold : int, optional
        Number of folds to use in stratified k fold. The default is 3.
    seed : int, optional
        Seed used for shuffling the train test split and stratified k fold.
        The default is None.
    undersample : bool, optional
        Whether or not to undersample the training data to get balanced data.

    Returns
    -------
    train_data : numpy array
        DESCRIPTION.
    train_labels : numpy array
        DESCRIPTION.
    train_ind : numpy array
        DESCRIPTION.
    val_ind : numpy array
        DESCRIPTION.
    test_data : numpy array
        DESCRIPTION.
    test_labels : numpy array
        DESCRIPTION.
    """

    data = sio.loadmat(filename)
    pressure = data['tactile_data']
    imu = data['IMU_data']
    object_id = data['object_id']
    
    pressure_imu = []
    for i in range(len(imu)):
        pressure_imu.append([pressure[i], imu[i]])

    if kfold is not None:
        # Decrease the test size if cross validation is used
        test_size = 0.15
    else:
        kfold = 3
        test_size = 0.05

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
                          random_state=seed+1)
    # train_ind, val_ind = skf.split(train_data, train_labels)
    # skf_gen = skf.split(train_data, train_labels)
    
    return train_data, train_labels, test_data, test_labels, skf
    