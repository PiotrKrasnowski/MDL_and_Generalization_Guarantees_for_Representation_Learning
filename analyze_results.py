import numpy as np
import matplotlib.pyplot as plt
import os

timestamp = '20230101-115959'

results_dir = 'results/results_' + timestamp + '/'
figures_dir = 'figures/figures_' + timestamp + '/'

filenames = os.listdir(results_dir)
global_parameters = np.load(results_dir + 'global_parameters.npz', allow_pickle=True)["global_parameters"][()]
filenames.remove('global_parameters.npz')

##############
# Statistics #
##############
accuracies_train = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_test  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_confidence_high_train = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_confidence_high_test  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_confidence_low_train = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_confidence_low_test  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))

log_likelihood_train = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
log_likelihood_test  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
log_likelihood_confidence_high_train = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]),global_parameters["repetitions"]))
log_likelihood_confidence_high_test  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]),global_parameters["repetitions"]))
log_likelihood_confidence_low_train = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]),global_parameters["repetitions"]))
log_likelihood_confidence_low_test  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]),global_parameters["repetitions"]))

izy_bound_train = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
izy_bound_test  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))

izx_bound_train = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
izx_bound_test  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))

accuracies_train_worst = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_test_worst  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_confidence_high_train_worst = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_confidence_high_test_worst  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_confidence_low_train_worst = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
accuracies_confidence_low_test_worst  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))

log_likelihood_train_worst = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
log_likelihood_test_worst  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
log_likelihood_confidence_high_train_worst = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
log_likelihood_confidence_high_test_worst  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
log_likelihood_confidence_low_train_worst = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))
log_likelihood_confidence_low_test_worst  = np.zeros((len(global_parameters["loss_id"]), len(global_parameters["beta"]), global_parameters["repetitions"]))

for file in filenames:
    if file[:9] != '_results_': continue
    data = np.load(results_dir + file, allow_pickle=True)
    solver_parameters = data["solver_parameters"][()]
    train1_train_dataset = data["train1_train_dataset"][()]
    train1_test_dataset  = data["train1_test_dataset"][()]
    train2_train_dataset = data["train2_train_dataset"][()]
    train2_test_dataset  = data["train2_test_dataset"][()]

    beta_id = global_parameters["beta"].index(solver_parameters["beta"])
    loss_id = global_parameters["loss_id"].index(solver_parameters["loss_id"])
    rptt_id = data["counter"]

    accuracies_train[loss_id, beta_id, rptt_id] += train1_train_dataset["accuracy"]
    accuracies_test[loss_id, beta_id, rptt_id] += train1_test_dataset["accuracy"]

    accuracies_confidence_high_train[loss_id, beta_id, rptt_id] += train1_train_dataset["accuracy_confidence_high"]
    accuracies_confidence_high_test[loss_id, beta_id, rptt_id] += train1_test_dataset["accuracy_confidence_high"]

    accuracies_confidence_low_train[loss_id, beta_id, rptt_id] += train1_train_dataset["accuracy_confidence_low"]
    accuracies_confidence_low_test[loss_id, beta_id, rptt_id] += train1_test_dataset["accuracy_confidence_low"]

    log_likelihood_train[loss_id, beta_id, rptt_id] += train1_train_dataset["log_likelihood"]
    log_likelihood_test[loss_id, beta_id, rptt_id] += train1_test_dataset["log_likelihood"]

    log_likelihood_confidence_high_train[loss_id, beta_id, rptt_id] += train1_train_dataset["log_likelihood_confidence_high"]
    log_likelihood_confidence_high_test[loss_id, beta_id, rptt_id] += train1_test_dataset["log_likelihood_confidence_high"]

    log_likelihood_confidence_low_train[loss_id, beta_id, rptt_id] += train1_train_dataset["log_likelihood_confidence_low"]
    log_likelihood_confidence_low_test[loss_id, beta_id, rptt_id] += train1_test_dataset["log_likelihood_confidence_low"]

    izy_bound_train[loss_id, beta_id, rptt_id] += train1_train_dataset["izy_bound"]
    izy_bound_test[loss_id, beta_id, rptt_id] += train1_test_dataset["izy_bound"]

    izx_bound_train[loss_id, beta_id, rptt_id] += train1_train_dataset["izx_bound"]
    izx_bound_test[loss_id, beta_id, rptt_id] += train1_test_dataset["izx_bound"]

    if solver_parameters["alpha"]:
        accuracies_train_worst[loss_id, beta_id, rptt_id] = train2_train_dataset["accuracy"][0]
        accuracies_test_worst[loss_id, beta_id, rptt_id] = train2_test_dataset["accuracy"][0]

        accuracies_confidence_high_train_worst[loss_id, beta_id, rptt_id] = train2_train_dataset["accuracy_confidence_high"][0]
        accuracies_confidence_high_test_worst[loss_id, beta_id, rptt_id] = train2_test_dataset["accuracy_confidence_high"][0]

        accuracies_confidence_low_train_worst[loss_id, beta_id, rptt_id] = train2_train_dataset["accuracy_confidence_low"][0]
        accuracies_confidence_low_test_worst[loss_id, beta_id, rptt_id] = train2_test_dataset["accuracy_confidence_low"][0]

        log_likelihood_train_worst[loss_id, beta_id, rptt_id] = train2_train_dataset["log_likelihood"][0]
        log_likelihood_test_worst[loss_id, beta_id, rptt_id] = train2_test_dataset["log_likelihood"][0]

        log_likelihood_confidence_high_train_worst[loss_id, beta_id, rptt_id] = train2_train_dataset["log_likelihood_confidence_high"][0]
        log_likelihood_confidence_high_test_worst[loss_id, beta_id, rptt_id] = train2_test_dataset["log_likelihood_confidence_high"][0]

        log_likelihood_confidence_low_train_worst[loss_id, beta_id, rptt_id] = train2_train_dataset["log_likelihood_confidence_low"][0]
        log_likelihood_confidence_low_test_worst[loss_id, beta_id, rptt_id] = train2_test_dataset["log_likelihood_confidence_low"][0]


###########
# Figures #
###########

# some options
label_format = '{:.0e}'
plt.rcParams.update({'font.size': 12})
Labels_Train = ['standard VIB, train', 'lossy CDVIB, train', 'lossless CDVIB, train']
Labels_Test  = ['standard VIB', 'lossy CDVIB', 'lossless CDVIB']
colors_Train = [':ro', ':b^', ':gs']
colors_Test  = ['-ro', '-b^', '-gs']
colors_Intervals = ['r', 'b', 'g']

fig_conf, ax1_conf = plt.subplots(1, 1)   
for loss_id in range(len(global_parameters["loss_id"])):             
    ax1_conf.plot(global_parameters["beta"], np.mean(accuracies_train[loss_id,:,:], axis=1), colors_Train[loss_id], label=Labels_Train[loss_id])
    ax1_conf.plot(global_parameters["beta"], np.mean(accuracies_test[loss_id,:,:], axis=1), colors_Test[loss_id], label=Labels_Test[loss_id])
    ax1_conf.fill_between(global_parameters["beta"], np.mean(accuracies_confidence_low_train[loss_id,:,:], axis=1), np.mean(accuracies_confidence_high_train[loss_id,:,:], axis=1), color=colors_Intervals[loss_id], alpha = 0.5)
    ax1_conf.fill_between(global_parameters["beta"], np.mean(accuracies_confidence_low_test[loss_id,:,:], axis=1), np.mean(accuracies_confidence_high_test[loss_id,:,:], axis=1), color=colors_Intervals[loss_id], alpha = 0.5)

ax1_conf.set_ylabel('Accuracy')
ax1_conf.set_xlabel('Beta')
ax1_conf.legend()
ax1_conf.grid()
ax1_conf.set_title('Accuracy')
ax1_conf.set_xscale('log')
# plt.xlim([-0.1, 1.1])

fig_conf.savefig(figures_dir + 'accuracy')

fig_conf, ax2_conf = plt.subplots(1, 1)
for loss_id in range(len(global_parameters["loss_id"])):              
    ax2_conf.plot(global_parameters["beta"], np.mean(log_likelihood_train[loss_id,:,:], axis=1), colors_Train[loss_id], label=Labels_Train[loss_id])
    ax2_conf.plot(global_parameters["beta"], np.mean(log_likelihood_test[loss_id,:,:], axis=1), colors_Test[loss_id], label=Labels_Test[loss_id])
    ax2_conf.fill_between(global_parameters["beta"], np.mean(log_likelihood_confidence_low_train[loss_id,:,:], axis=1), np.mean(log_likelihood_confidence_high_train[loss_id,:,:], axis=1), color=colors_Intervals[loss_id], alpha = 0.5)
    ax2_conf.fill_between(global_parameters["beta"], np.mean(log_likelihood_confidence_low_test[loss_id,:,:], axis=1), np.mean(log_likelihood_confidence_high_test[loss_id,:,:], axis=1), color=colors_Intervals[loss_id], alpha = 0.5)

ax2_conf.set_ylabel('Log likelihood')
ax2_conf.set_xlabel('Beta')
ax2_conf.legend()
ax2_conf.grid()
ax2_conf.set_title('Log likelihood')
ax2_conf.set_xscale('log')

fig_conf.savefig(figures_dir + 'log_likelihood')

fig_conf, ax3_conf = plt.subplots(1, 1)
for loss_id in range(len(global_parameters["loss_id"])):                  
    ax3_conf.plot(global_parameters["beta"], np.mean(log_likelihood_train_worst[loss_id,:,:], axis=1), label=Labels_Train[loss_id])
    ax3_conf.plot(global_parameters["beta"], np.mean(log_likelihood_test_worst[loss_id,:,:], axis=1), colors_Test[loss_id], label=Labels_Test[loss_id])
    ax3_conf.fill_between(global_parameters["beta"], np.mean(log_likelihood_confidence_low_train_worst[loss_id,:,:], axis=1), np.mean(log_likelihood_confidence_high_train_worst[loss_id,:,:], axis=1), color=colors_Intervals[loss_id], alpha = 0.5)
    ax3_conf.fill_between(global_parameters["beta"], np.mean(log_likelihood_confidence_low_test_worst[loss_id,:,:], axis=1), np.mean(log_likelihood_confidence_high_test_worst[loss_id,:,:], axis=1), color=colors_Intervals[loss_id], alpha = 0.5)

ax3_conf.set_ylabel('Log likelihood')
ax3_conf.set_xlabel('Beta')
ax3_conf.legend()
ax3_conf.grid()
ax3_conf.set_title('Log likelihood, "worst" decoder')
ax3_conf.set_xscale('log')

fig_conf.savefig(figures_dir + 'log_likelihood_worst_decoder')

fig_conf, ax4_conf = plt.subplots(1, 1)
for loss_id in range(len(global_parameters["loss_id"])):                 
    ax4_conf.plot(global_parameters["beta"], np.mean(accuracies_train_worst[loss_id,:,:], axis=1), label=Labels_Train[loss_id])
    ax4_conf.plot(global_parameters["beta"], np.mean(accuracies_test_worst[loss_id,:,:], axis=1), colors_Test[loss_id], label=Labels_Test[loss_id])
    ax2_conf.fill_between(global_parameters["beta"], np.mean(accuracies_confidence_low_train_worst[loss_id,:,:], axis=1), np.mean(accuracies_confidence_high_train_worst[loss_id,:,:], axis=1), color=colors_Intervals[loss_id], alpha = 0.5)
    ax2_conf.fill_between(global_parameters["beta"], np.mean(accuracies_confidence_low_test_worst[loss_id,:,:], axis=1), np.mean(accuracies_confidence_high_test_worst[loss_id,:,:], axis=1), color=colors_Intervals[loss_id], alpha = 0.5)

ax4_conf.set_ylabel('Accuracy')
ax4_conf.set_xlabel('Beta')
ax4_conf.legend()
ax4_conf.grid()
ax4_conf.set_title('Accuracy, "worst" decoder')
ax4_conf.set_xscale('log')
# plt.xlim([-0.1, 1.1])

fig_conf.savefig(figures_dir + 'accuracy_worst_decoder')
