import numpy as np
import matplotlib.pyplot as plt


def compute_network_performance(results_=None, rg_inputoutput=None, issue_free=False, make_plots=False):
    performance_metrics = []
    nof_runs = len(results_['df_results'])

    for r in range(nof_runs):
        pred_acc = []
        pred_perf = []
        pred_perf_per_string = []

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

        metrics = {
            'prediction_accuracy': pred_acc,
            'prediction_performance': pred_perf,
            'prediction_performance_per_string': pred_perf_per_string
        }
        performance_metrics.append(metrics)

        if make_plots:
            # ____________________________PPR and PAR PLOTS_________________________________________
            plt.figure(1, figsize=(25, 7 * nof_runs))
            plt.subplot(nof_runs, 1, r + 1)
            plt.plot(pred_perf, label='PPR', color='brown', lw=2)
            plt.plot(pred_acc, label='PAR', color='darkcyan', lw=2)
            plt.ylabel('PPR, PAR Score (in %)', fontsize=25)
            plt.yticks(fontsize=25)
            plt.xlabel('Timestep', fontsize=25)
            plt.xticks(fontsize=25)
            plt.legend(loc='lower center', facecolor='lightgrey', fontsize=25)
            plt.grid(True, linestyle="--", color='black', alpha=0.4)
            # plt.title(f'Mean prediction accuracy (in %): {np.mean([pred_[0] for pred_ in pred_acc])}')

            # _____________Frequency bar_________________
            plt.figure(2, figsize=(12, 7 * nof_runs))
            plt.subplot(nof_runs, 1, r + 1)
            plt.hist([pred_perf, pred_acc], bins=np.arange(0, 110, 10), color=['brown', 'darkcyan'], label=['PPR', 'PAR'])
            plt.ylabel('Frequency of \n PPR / PAR score', fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('PPR / PAR Score', fontsize=20)
            plt.xticks(np.arange(0, 110, 10), fontsize=20)
            plt.legend(loc='upper left', facecolor='lightgrey', fontsize=20)
            plt.grid(True, linestyle="--", color='black', alpha=0.4)

            # _____________P3S PLOTS_____________________
            plt.figure(3, figsize=(25, 7 * nof_runs))
            plt.subplot(nof_runs, 1, r + 1)
            plt.plot(pred_perf_per_string, label='PPR', color='brown', lw=2)
            plt.ylabel('P3S Score (in %)', fontsize=25)
            plt.yticks(fontsize=25)
            plt.xlabel('String Index', fontsize=25)
            plt.xticks(fontsize=25)
            plt.grid(True, linestyle="--", color='black', alpha=0.4)

            plt.show()
            plt.close()

    return np.array(performance_metrics, dtype=object)


def compute_network_performance_averages(performance_metrics=None, ma_len=100, make_plots=False):

    nof_runs = len(performance_metrics)
    nof_strings = len(performance_metrics[0]['prediction_performance_per_string'])

    # __________________Computing Mean and SD of P3S score_______________________________________
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

    # _____________Computing moving averages of PAR and PPR scores per run________________________
    moving_average_par_per_run = []
    moving_average_ppr_per_run = []

    for r in range(nof_runs):
        MA_pred_acc = []
        MA_pred_perf = []
        i = 0
        while i + ma_len < len(performance_metrics[r]['prediction_performance']):
            MA_pred_acc.append(
                np.mean([acc_ for acc_ in performance_metrics[r]['prediction_accuracy'][i:i + ma_len]])
            )
            MA_pred_perf.append(
                np.mean([perf_ for perf_ in performance_metrics[r]['prediction_performance'][i:i + ma_len]])
            )
            i += 1
        moving_average_par_per_run.append(MA_pred_acc)
        moving_average_ppr_per_run.append(MA_pred_perf)

    moving_average_par_per_run = np.array(moving_average_par_per_run)
    moving_average_ppr_per_run = np.array(moving_average_ppr_per_run)

    # _________Computing mean and SD of the moving averages of PAR and PPR scores per run_____________
    len_inputstream = np.min([len(moving_average_par_per_run[r]) for r in range(nof_runs)])

    avg_moving_average_ppr = []
    sd_moving_average_ppr = []
    avg_moving_average_par = []
    sd_moving_average_par = []
    for step in range(len_inputstream):
        avg_moving_average_ppr.append(
            np.mean([moving_average_ppr_per_run[r][step] for r in range(nof_runs)])
        )
        sd_moving_average_ppr.append(
            np.std([moving_average_ppr_per_run[r][step] for r in range(nof_runs)])
        )

        avg_moving_average_par.append(
            np.mean([moving_average_par_per_run[r][step] for r in range(nof_runs)])
        )
        sd_moving_average_par.append(
            np.std([moving_average_par_per_run[r][step] for r in range(nof_runs)])
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

    if make_plots:
        # PLOTTING THE AVERAGE P3S SCORE
        plt.figure(5, figsize=(28, 8 * 2))
        plt.subplot(2, 1, 1)
        plt.plot(avg_prediction_performance_per_string, label='P3S', color='darkgreen', lw=2)
        plt.fill_between([i for i in range(nof_strings)],
                         avg_prediction_performance_per_string - sd_prediction_performance_per_string,
                         avg_prediction_performance_per_string + sd_prediction_performance_per_string,
                         color='green', alpha=0.2)
        plt.ylabel('Average P3S Score (in %)', fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('String Index', fontsize=25)
        plt.xticks(fontsize=25)
        plt.grid(True, linestyle="--", color='black', alpha=0.6)

        # PLOTTING THE AVERAGE MAs OF PAR AND PPR SCORES
        plt.subplot(2, 1, 2)
        plt.plot(avg_moving_average_ppr, label='Moving Average of PPR', color='brown', lw=2)
        plt.fill_between([i for i in range(len_inputstream)],
                         avg_moving_average_ppr - sd_moving_average_ppr,
                         avg_moving_average_ppr + sd_moving_average_ppr,
                         color='magenta', alpha=0.2)
        plt.plot(avg_moving_average_par, label='Moving Average of PAR', color='darkcyan', lw=2)
        plt.fill_between([i for i in range(len_inputstream)],
                         avg_moving_average_par - sd_moving_average_par,
                         avg_moving_average_par + sd_moving_average_par,
                         color='cyan', alpha=0.2)
        plt.ylabel('Moving Average of \n PPR, PAR Score over \n 100 timesteps (in %)', fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Timestep', fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(loc='lower center', facecolor='lightgrey', fontsize=25)
        plt.grid(True, linestyle="--", color='black', alpha=0.6)

        plt.show()
        plt.close()

    return performance_metrics_average
