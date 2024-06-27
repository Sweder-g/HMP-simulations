import hmp
from hmp import simulations
import numpy as np
import scipy.stats
import os

# Frequency of the event defining its duration, half-sine of 10Hz = 50ms
# Amplitude of the event in nAm, defining signal to noise ratio
# location, extent and source parameters according to the MNE package
adjustable_pars = {
    "frequency": 10.,
    "location": 'center',
    "extent": 0.0,
    "source": [0, 0, 0]
}

def generate_participant(filename, n_trials, amplitude):
    """
    Function to generate a single participant given a parameter and a normal distribution

    Parameters
    ----------
    filename (string): the name for the participant files
    n_trials (int): the number of trials that will be simulated 
    amplitude (float): the amplitude for the current dataset
    """

    # parameters for the simulation process (based on the tutorial from Weindel et al.)
    cpus = 4  # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM
    sfreq = 100
    n_events = 4
    shape = 2  # shape of the gamma distribution
    means = np.array([60, 150, 200, 100, 80]) / shape  # Mean duration of the between event times in ms
    names = ['inferiortemporal-lh', 'caudalanteriorcingulate-rh', 'bankssts-lh', 'superiorparietal-lh', 'superiorparietal-lh']  # Which source to activate for each event (see atlas when calling simulations.available_sources())
    sources = []

    # Create sources
    for source in zip(names, means):  # One source = one frequency, one amplitude and a given by-trial variability distribution
        sources.append([source[0], adjustable_pars["frequency"], amplitude, scipy.stats.gamma(shape, scale=source[1])])

    # Generate the data with the given parameters
    simulations.simulate(sources, n_trials, cpus, filename, overwrite=False, sfreq=sfreq, seed=1234, a=adjustable_pars["location"], b=adjustable_pars["extent"], x=adjustable_pars["source"][0], y=adjustable_pars["source"][1], z=adjustable_pars["source"][2])

def generate_dataset(n_participants, base_folderpath, n_trials, amplitudes):
    """
    Function that uses generate_participant() to generate a dataset of multiple participants for each amplitude

    Parameters
    ----------
    n_participants (int): number of participants that will be simulated
    base_folderpath (string): the base path for the participant files
    n_trials (int): the number of trials that will be simulated 
    amplitudes (list of floats): the list of amplitudes for which datasets will be generated
    """
    for amplitude in amplitudes:
        folder_name = f"{n_participants} participants {n_trials} trials with amplitude {amplitude}"
        folderpath = os.path.join(base_folderpath, folder_name)
        os.makedirs(folderpath, exist_ok=True)
        filename = os.path.join(folderpath, folder_name)
        for participant in range(n_participants):
            participant_filename = f"{filename}_{participant}.h5"
            generate_participant(participant_filename, n_trials, amplitude)

# Get user input
n_participants = int(input("How many subjects do you want to simulate: "))
n_trials = int(input("How many trials do you want to simulate: "))

#Enter a list of amplitudes for which the data will be generated
amplitudes = []

base_folderpath = "c:\\Users\\Sweder\\path\\to\\data"

# Generate the datasets
generate_dataset(n_participants, base_folderpath, n_trials, amplitudes)