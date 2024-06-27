import os
import hmp
import numpy as np
import re
import glob
import pandas as pd
import csv
import matplotlib.pyplot as plt

def get_files(path):
    """
    function that returns arrays of .fif, .npy, .nc (with or without true_model classification) files found in a given folder

    Parameters
    ----------
    folder (string): path to the folder where the files are found

    Returns
    -------
    returns (fif_files, npy_files, nc_files, true_fif, true_npy)
    """
    # Set path
    folder = os.listdir(path)
    print("Contents of the directory:", folder)

    # Append files to arrays
    fif_files = []
    npy_files = []

    for file in folder:
        file_path = os.path.join(path, file)
        if file.endswith("fif"):
            fif_files.append(file_path)
        elif file.endswith("events.npy"):
            npy_files.append(file_path)

    return fif_files, npy_files


def epoch_data(fif_files, npy_files):
    """
    function that creates epoched data from .fif and .npy files

    Parameters
    ----------
    fif_files (string[]): an array that contains all the names of the .fif files 
    npy_files (string[]): an array that contains all the names of the .npy files 

    returns
    -------
    epoch_data (list of xarray.Dataset): the epoched data
    events_matrix (ndarray[]): array with all the event matrices
    source_times (float[]): array with all the source times
    """

    epoch_data = []
    events_matrix = []
    source_times = []
    sfreq = 100

    # read files as EEG data
    for i in range(len(fif_files)): #for every participant..

        # npy
        events = np.load(npy_files[i]) #load the events from the .npy file..
        resp_trigger = int(np.max(np.unique(events[:, 2])))  # Resp trigger is the last source in each trial
        event_id = {'stimulus': 1}  # trigger 1 = stimulus
        resp_id = {'response': resp_trigger}

        # fif
        eeg_data = hmp.utils.read_mne_data(fif_files[i], event_id=event_id, resp_id=resp_id, sfreq=sfreq, events_provided=events, verbose=False)

        # source_times
        n_trials = len(eeg_data["epochs"])
        source_time = np.mean(np.reshape(np.diff(events[:, 0], prepend=0), (n_trials, 5 + 1))[:, 1:])  # By-trial generated event times

        epoch_data.append(eeg_data)
        events_matrix.append(events)
        source_times.append(source_time)

    # Convert lists to appropriate data types
    events_matrix = np.array(events_matrix)
    source_times = np.array(source_times)

    return epoch_data, events_matrix, source_times


def fit_model_better(epoch_data=None, participants_variable="participant", apply_standard=False, averaged=False, apply_zscore='trial', zscore_acrossPCs=False, method='pca', cov=True, centering=True, n_comp=None, n_ppcas=None, pca_weights=None, bandfilter=None, mcca_reg=0, cpus=1, save=None, filename=None):
    """
    Function that fits an HMP model

    Parameters
    ----------
    epoch_data : xarray
        Epoched data: unstacked xarray data from transform_data() or any other source yielding an xarray with dimensions 
        [participant * epochs * samples * channels]
    participants_variable : str
        Name of the dimension for participants ID
    apply_standard : bool 
        Whether to apply standardization of variance between participants, recommended when they are few of them (e.g. < 10)
    averaged : bool
        Applying the pca on the averaged ERP (True) or single trial ERP (False, default). No effect if cov = True
    apply_zscore : str 
        Whether to apply z-scoring and on what data, either None, 'all', 'participant', 'trial', for zscoring across all data, by participant, or by trial, respectively. If set to true, evaluates to 'trial' for backward compatibility.
    method : str
        Method to apply, 'pca' or 'mcca'
    cov : bool
        Whether to apply the pca/mcca to the variance covariance (True, default) or the epoched data
    n_comp : int
        How many components to select from the PC space, if None plots the scree plot and a prompt requires user
        to specify how many PCs should be retained
    n_ppcas : int
        If method = 'mcca', controls the number of components retained for the by-participant PCAs
    pca_weights : xarray
        Weights of a PCA to apply to the data (e.g. in the resample function)
    bandfilter: None | (lfreq, hfreq) 
        If none, no filtering is applied. If tuple, data is filtered between lfreq-hfreq.
        NOTE: filtering at this step is suboptimal, filter before epoching if at all possible, see
              also https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
    mcca_reg: float
        Regularization used for the mcca computation (see mcca.py)
    filename : str
        Name of the file to save the HMP-model
    save : bool
        If true, will save the HMP-model

    Returns
    -------
    Selected (hmp): the predicted events of the HMP-model
    Init (hmp): the initialization of the HMP-model
    """
    print("Initializing a hmp model using " + str(method) + "...")

    hmp_dat = hmp.utils.transform_data(data=epoch_data, averaged=averaged, method=method, cov=cov, n_comp=n_comp, n_ppcas=n_ppcas, mcca_reg=mcca_reg)

    # Initialization of the model
    init = hmp.models.hmp(hmp_dat, sfreq=epoch_data.sfreq, event_width=50, cpus=cpus)

    # Fitting
    print("fitting...")
    selected = init.fit()  # function to fit an instance of a x events model

    return selected, init