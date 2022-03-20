import numpy as np


def compute_network_performance(results_=None, rg_inputoutput=None, ma_len=100, issue_free=False):
    performance_metrics = []

    for r in range(len(results_['df_results'])):
        pred_acc = []
        pred_perf = []
        pred_perf_per_string = []
        MA_pred_acc = []
        MA_pred_perf = []

        if issue_free:
            df_results = results_['df_results'][r]
            df_results = df_results[df_results['issue'] == 'none']
        else:
            df_results = results_['df_results'][r]

        rg_inputoutput_strings = np.array([c[0] for c in rg_inputoutput])

        for idx, string in enumerate(df_results.index):
            string_predictions = df_results.loc[string]['htm_preds']
            rg_io = rg_inputoutput[np.where(rg_inputoutput_strings == string)[0][0]]

            for step in range(len(string_predictions)):
                correct_predicted_cols_idx = np.where(rg_io[2][step])[0] # Correct expected predictions for the
                # particular <[string_idx][step]> time step

                predicted_cols_idx = np.unique(np.where(string_predictions[step])[1]) # Indices of the cols predicted by
                # the network for particular <[string_idx][step]> time step

                count = 0
                for col_idx in correct_predicted_cols_idx:
                    if col_idx in predicted_cols_idx:
                        count += 1
                if len(predicted_cols_idx) == 0:
                    accuracy = 0.0
                else:
                    accuracy = (count / len(predicted_cols_idx)) * 100

                performance = (count / len(correct_predicted_cols_idx)) * 100

                pred_acc.append(accuracy)
                pred_perf.append(performance)

            pred_perf_per_string.append(np.mean(pred_perf[-len(string_predictions):]))

        i = 0
        while i + ma_len < len(pred_perf):
            MA_pred_acc.append(np.mean([acc_ for acc_ in pred_acc[i:i + ma_len]]))
            MA_pred_perf.append(np.mean([perf_ for perf_ in pred_perf[i:i + ma_len]]))
            i += 1

        metrics = {
            'prediction_accuracy': pred_acc,
            'prediction_performance': pred_perf,
            'prediction_performance_per_string': pred_perf_per_string,
            'moving_average_accuracy': MA_pred_acc,
            'moving_average_performance': MA_pred_perf,
        }
        performance_metrics.append(metrics)

    return np.array(performance_metrics, dtype=object)


def compute_network_performance_averages(performance_metrics=None):

    nof_runs = len(performance_metrics)
    len_inputstream = np.min([len(performance_metrics[r]['moving_average_accuracy']) for r in range(nof_runs)])
    nof_strings = len(performance_metrics[0]['prediction_performance_per_string'])

    avg_prediction_performance_per_string = []
    sd_prediction_performance_per_string = []
    for st_ in range(nof_strings):
        avg_prediction_performance_per_string.append(
            np.mean([performance_metrics[r]['prediction_performance_per_string'][st_] for r in range(nof_runs)])
        )
        sd_prediction_performance_per_string.append(
            np.std([performance_metrics[r]['prediction_performance_per_string'][st_] for r in range(nof_runs)])
        )

    avg_prediction_performance_per_string = np.array(avg_prediction_performance_per_string)
    sd_prediction_performance_per_string = np.array(sd_prediction_performance_per_string)

    avg_moving_average_ppr = []
    sd_moving_average_ppr = []
    avg_moving_average_par = []
    sd_moving_average_par = []
    for step in range(len_inputstream):
        avg_moving_average_ppr.append(
            np.mean([performance_metrics[r]['moving_average_performance'][step] for r in range(nof_runs)])
        )
        sd_moving_average_ppr.append(
            np.std([performance_metrics[r]['moving_average_performance'][step] for r in range(nof_runs)])
        )

        avg_moving_average_par.append(
            np.mean([performance_metrics[r]['moving_average_accuracy'][step] for r in range(nof_runs)])
        )
        sd_moving_average_par.append(
            np.std([performance_metrics[r]['moving_average_accuracy'][step] for r in range(nof_runs)])
        )

    avg_moving_average_ppr = np.array(avg_moving_average_ppr)
    sd_moving_average_ppr = np.array(sd_moving_average_ppr)
    avg_moving_average_par = np.array(avg_moving_average_par)
    sd_moving_average_par = np.array(sd_moving_average_par)

    performance_metrics_average = {
        'avg_prediction_performance_per_string': avg_prediction_performance_per_string,
        'sd_prediction_performance_per_string': sd_prediction_performance_per_string,
        'avg_moving_average_ppr': avg_moving_average_ppr,
        'sd_moving_average_ppr': sd_moving_average_ppr,
        'avg_moving_average_par': avg_moving_average_par,
        'sd_moving_average_par': sd_moving_average_par
    }

    return performance_metrics_average
