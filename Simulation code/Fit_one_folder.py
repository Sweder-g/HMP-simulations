import os
import hmp
import numpy as np
import matplotlib.pyplot as plt
from hmp import simulations
import General_Functions as general  # type: ignore
import multiprocessing as mp
import xarray as xr
import pandas as pd

# 0. Initialization
# 0 get fif and npy files
folder = "40 participants 50 trials with amplitude 1e-08"
path = "c:\\Users\\Sweder\\Desktop\\Generated data\\" + folder + "\\"

# Extract amplitude from the folder string
amplitude_str = folder.split(" with amplitude ")[-1]
amplitude = float(amplitude_str)

fif_files, npy_files = general.get_files(path)

signalless_fif, signalless_events = general.get_files("c:\\Users\\Sweder\\Desktop\\signalless_fif_and_events\\")  # get signalless files to get average for noise

# 1. Epoch data
epoch_data, events_matrix, source_times = general.epoch_data(fif_files, npy_files)
SL_epoch_data, SL_events_matrix, SL_source_times = general.epoch_data(signalless_fif, signalless_events)

# 2. Set parameters (based on the tutorial from Weindel et al.)
cpus = 1  # 1 seems to be the only value for which the code works, 2 or more cpus give errors
mp.set_start_method("spawn")

resp_trigger = int(np.max(np.unique(events_matrix[0][:, 2])))  # resp trigger is the same for all indexes of events_matrix so we just take it from events_matrix[0]
SL_resp_trigger = int(np.max(np.unique(SL_events_matrix[0][:, 2])))

event_id = {'stimulus': 1}  # trigger 1 = stimulus
resp_id = {'response': resp_trigger}
SL_resp_id = {'response': SL_resp_trigger}
sfreq = 100
tmin, tmax = -.25, 2  # window size for the epochs, from 250ms before the stimulus up to 2 seconds after, data will be baseline corrected from tmin to 0
high_pass = 1  # High pass filtering to be applied, useful to avoid slow drifts, parameters of the filters are defined automatically by MNE

epoch_data = hmp.utils.read_mne_data(fif_files, event_id=event_id, resp_id=resp_id, epoched=False, tmin=tmin, tmax=tmax, sfreq=sfreq, high_pass=high_pass, events_provided=events_matrix, verbose=False)

epoch_signalless_data = hmp.utils.read_mne_data(signalless_fif, event_id=event_id, resp_id=SL_resp_id, epoched=False, tmin=tmin, tmax=tmax, sfreq=sfreq, high_pass=high_pass, events_provided=SL_events_matrix, verbose=False)

# Calculate the standard deviation for each electrode (channel) over all samples and trials
data = epoch_signalless_data["data"].values
signalless_sdevs_per_channel = np.nanstd(data, axis=(1, 3))

print("Standard Deviation per signalless channel: ", signalless_sdevs_per_channel)

print("mean sdev of all channels: ", np.mean(signalless_sdevs_per_channel))

n_comp = 5
n_ppcas = 15

strategies = [
    {"method": "pca", "varcov_or_erp": "vc"},
    {"method": "pca", "varcov_or_erp": "erp"},
    {"method": "mcca", "varcov_or_erp": "vc"},
    {"method": "mcca", "varcov_or_erp": "erp"}
]

results = []

for strategy in strategies:
    method = strategy["method"]
    varcov_or_erp = strategy["varcov_or_erp"]

    if varcov_or_erp == "erp":
        averaged = True  # apply the pca on averaged erp
        cov = False  # apply the pca/mcca to the epoched data
    else:
        averaged = False  # apply the pca on averaged single trial erp (no effect if cov = True)
        cov = True  # apply the pca/mcca to the variance covariance (True, default)

    savedfilename = f"{method}_method_on_{varcov_or_erp}_using_{n_comp}_PCs"

    # 3. Fit the chosen model
    predicted_events, init = general.fit_model_better(
        epoch_data=epoch_data,
        participants_variable="participant",
        apply_standard=False,
        averaged=averaged,
        apply_zscore='trial',
        zscore_acrossPCs=False,
        method=method,
        cov=cov,
        centering=True,
        n_comp=n_comp,
        n_ppcas=n_ppcas,
        pca_weights=None,
        bandfilter=None,
        mcca_reg=0,
        cpus=cpus,
        save='yes',
        filename=os.path.join(path, savedfilename)
    )

    # for the "true" model: stack the events of the event matrices of all participants
    events_stack_true_model = np.vstack(events_matrix)

    true_sim_source_times, true_pars, true_magnitudes, _ = simulations.simulated_times_and_parameters(events_stack_true_model, init)

    # We do that by calling fit_single on init, and telling it to recover 4 events and using the true parameters and magnitudes as starting points. We also tell it that it does not in fact have to maximize the fit.
    n_events = 4
    true_estimates = init.fit_single(n_events, parameters=true_pars, magnitudes=true_magnitudes, maximization=False)

    # 4. Getting topologies of fitted model
    test_topologies = init.compute_topologies(epoch_data, predicted_events, init, mean=True)  # we get the values of each electrode for the peak of each event the model finds
    true_topologies = init.compute_topologies(epoch_data, true_estimates, init, mean=True)

    # 4.1 getting the signal-to-noise ratio for the chosen strategy
    # Compute the absolute values of the topologies
    test_topologies_absolute = np.abs(test_topologies)

    # Compute the signal-to-noise ratio (SNR)
    snr_forall_electrodes_and_events = test_topologies_absolute / signalless_sdevs_per_channel  # element-wise division

    mean_snr_per_channel = np.mean(snr_forall_electrodes_and_events, axis=0)
    mean_snr = np.mean(snr_forall_electrodes_and_events)
    mean_snr_rounded = float(mean_snr.round(4))
    print(mean_snr_rounded)

    # Calculate TPR and PPV
    idx_true_positive, corresp_true_idx = simulations.classification_true(true_topologies, test_topologies)

    TP = len(idx_true_positive)
    FN = len(true_estimates['event']) - TP
    FP = len(predicted_events['event']) - TP
    TPR = TP / (TP + FN)
    PPV = TP / (TP + FP)

    # 5. Plotting the topologies
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 5))
    fig.suptitle(folder)
    fig.subplots_adjust(hspace=0.4)

    title1 = f"Model estimate using {method.upper()} method on {varcov_or_erp} using {n_comp} PC's"
    title2 = f"Ground truth of using {method.upper()} method on {varcov_or_erp} using {n_comp} PC's"
    if method == "mcca":
        title1 += f" from {n_ppcas} PC's per participant"
        title2 += f" from {n_ppcas} PC's per participant"

    ax[0].set_title(title1)
    ax[1].set_title(title2)

    info = simulations.simulation_positions()

    hmp.visu.plot_topo_timecourse(epoch_data, predicted_events, info, init, magnify=1, sensors=True, times_to_display=np.mean(np.cumsum(true_sim_source_times, axis=1), axis=0), ax=ax[0], as_time=False)

    hmp.visu.plot_topo_timecourse(epoch_data, true_estimates, info, init, magnify=1, sensors=True, times_to_display=np.mean(np.cumsum(true_sim_source_times, axis=1), axis=0), ax=ax[1], as_time=False)

    # Add TPR and PPV to the plot
    tpr_text = f"TPR: {TPR:.2f}"
    ppv_text = f"PPV: {PPV:.2f}"
    ax[0].text(0.00, 0.95, tpr_text, transform=ax[0].transAxes, verticalalignment='top')
    ax[0].text(0.00, 0.85, ppv_text, transform=ax[0].transAxes, verticalalignment='top')

    # Save the plot as PNG
    plot_filename = os.path.join(path, f"{method}_{varcov_or_erp}_plot_with_amplitude_{amplitude_str}.png")
    plt.savefig(plot_filename)
    plt.close(fig)  # Close the figure to free up memory

    # Collect results
    results.append({
        "amplitude": amplitude,
        "method": method,
        "vc/erp": varcov_or_erp,
        "snr": mean_snr_rounded,
        "TPR": TPR,
        "PPV": PPV
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(path, "results.csv")
results_df.to_csv(results_csv_path, index=False)