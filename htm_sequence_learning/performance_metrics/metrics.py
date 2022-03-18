import numpy as np


def prediction_accuracy_ratio():
    return prediction_accuracy_ratio, moving_average_par


def prediction_performance_ratio():
    return prediction_performance_ratio, prediction_performance_perString, moving_average_ppr




for r in range(n_runs):

    results = list_results[r]
    rg_inputoutput = list_rg_inputoutput[r]

    pred_acc = []
    pred_perf = []
    pred_perf_perString = []
    MA_pred_acc = []
    MA_pred_perf = []
    for string_idx in range(nof_strings):

        performancePerString = []
        for step in range(len(results.iloc[string_idx]['htm_preds'])):

            correct_preds_cols_idx = np.where(rg_inputoutput[string_idx][2][step])[0]
            # Correct expected predictions for the particular <[string_idx][step]> timestep

            predicted_cols_idx = np.unique(np.where(results.iloc[string_idx]['htm_preds'][step])[1])
            # Indices of the cols predicted by the network for particular <[string_idx][step]> timestep.

            count = 0
            for col_idx in correct_preds_cols_idx:
                if col_idx in predicted_cols_idx:
                    count += 1

            if len(predicted_cols_idx) == 0:
                accuracy = 0.0
            else:
                accuracy = (count / len(predicted_cols_idx)) * 100

            performance = (count / len(correct_preds_cols_idx)) * 100

            pred_acc.append((accuracy, (string_idx, step)))
            pred_perf.append((performance, (string_idx, step)))
            performancePerString.append(performance)

        pred_perf_perString.append(np.mean(performancePerString))

    i = 0
    while i + 100 < len(pred_perf):
        MA_pred_acc.append(np.mean([acc_[0] for acc_ in pred_acc[i:i + 100]]))
        MA_pred_perf.append(np.mean([perf_[0] for perf_ in pred_perf[i:i + 100]]))
        i += 1

    list_pred_acc.append(pred_acc)
    list_pred_perf.append(pred_perf)
    list_pred_perf_perString.append(pred_perf_perString)
    list_MA_pred_acc.append(MA_pred_acc)
    list_MA_pred_perf.append(MA_pred_perf)

#______________________________________________________________________________________--
avg_p3s = []
sd_p3s = []
for st_ in range(nof_strings):
    avg_p3s.append(np.mean([list_pred_perf_perString[r][st_] for r in range(n_runs)]))
    sd_p3s.append(np.std([list_pred_perf_perString[r][st_] for r in range(n_runs)]))

avg_p3s = np.array(avg_p3s)
sd_p3s = np.array(sd_p3s)

avg_MAppr = []
sd_MAppr = []
avg_MApar = []
sd_MApar = []
for step in range(shortest_inputstream):
    avg_MAppr.append(np.mean([list_MA_pred_perf[r][step] for r in range(n_runs)]))
    sd_MAppr.append(np.std([list_MA_pred_perf[r][step] for r in range(n_runs)]))

    avg_MApar.append(np.mean([list_MA_pred_acc[r][step] for r in range(n_runs)]))
    sd_MApar.append(np.std([list_MA_pred_acc[r][step] for r in range(n_runs)]))

avg_MAppr = np.array(avg_MAppr)
sd_MAppr = np.array(sd_MAppr)
avg_MApar = np.array(avg_MApar)
sd_MApar = np.array(sd_MApar)