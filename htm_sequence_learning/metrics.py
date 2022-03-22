import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from htm_sequence_learning.visualizer_funcs import make_performance_plots, make_performance_averages_plots


def compute_network_performance(results_=None,
                                rg_inputoutput=None,
                                issue_free=False,
                                compute_averages=True,
                                make_plots=True,
                                save_figures=True,
                                fig_filename=None):
    performance_metrics = []
    performance_metrics_average = None
    nof_runs = len(results_['df_results'])

    for r in range(nof_runs):
        pred_acc = []
        pred_perf = []
        pred_perf_per_string = []

        if issue_free:
            df_results = pd.DataFrame(results_['df_results'][r],
                                      columns=['reber_string', 'htm_states', 'htm_preds', 'htm_pred_dendrites',
                                               'htm_winner_cells', 'num_net_dendrites', 'issue'])
            df_results = df_results[df_results['issue'] == 'none']
        else:
            df_results = pd.DataFrame(results_['df_results'][r],
                                      columns=['reber_string', 'htm_states', 'htm_preds', 'htm_pred_dendrites',
                                               'htm_winner_cells', 'num_net_dendrites', 'issue'])

        rg_inputoutput_strings = np.array([c[0] for c in rg_inputoutput])

        for idx, string in enumerate(df_results['reber_string']):
            string_predictions = df_results.loc[idx]['htm_preds']
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

            make_performance_plots(run_idx=r,
                                   pred_accuracy=pred_acc,
                                   pred_performance=pred_perf,
                                   pred_performance_per_string=pred_perf_per_string,
                                   save_figures=save_figures,
                                   fig_filename=fig_filename)

    performance_metrics = np.array(performance_metrics, dtype=object)

    if compute_averages:
        performance_metrics_average = compute_network_performance_averages(performance_metrics=performance_metrics,
                                                                           make_plots=make_plots,
                                                                           save_figures=save_figures,
                                                                           fig_filename=fig_filename)

    return performance_metrics, performance_metrics_average


def compute_network_performance_averages(performance_metrics=None,
                                         ma_len=100,
                                         make_plots=True,
                                         save_figures=True,
                                         fig_filename=None):

    nof_runs = len(performance_metrics)
    nof_strings = len(performance_metrics[0]['prediction_performance_per_string'])

    # ______________________Computing Mean and SD of P3S score___________________________________
    avg_prediction_performance_per_string = []
    sd_prediction_performance_per_string = []
    for st_ in range(nof_strings):
        avg_prediction_performance_per_string.append(
            np.mean([performance_metrics[r]['prediction_performance_per_string'][st_] for r in range(nof_runs)])
        )
        sd_prediction_performance_per_string.append(
            np.std([performance_metrics[r]['prediction_performance_per_string'][st_] for r in range(nof_runs)])
        )
    avg_prediction_performance_per_string = np.array(avg_prediction_performance_per_string, dtype=object)
    sd_prediction_performance_per_string = np.array(sd_prediction_performance_per_string, dtype=object)

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

    moving_average_par_per_run = np.array(moving_average_par_per_run, dtype=object)
    moving_average_ppr_per_run = np.array(moving_average_ppr_per_run, dtype=object)

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

    avg_moving_average_ppr = np.array(avg_moving_average_ppr, dtype=object)
    sd_moving_average_ppr = np.array(sd_moving_average_ppr, dtype=object)
    avg_moving_average_par = np.array(avg_moving_average_par, dtype=object)
    sd_moving_average_par = np.array(sd_moving_average_par, dtype=object)

    performance_metrics_average = {
        'avg_prediction_performance_per_string': avg_prediction_performance_per_string,
        'sd_prediction_performance_per_string': sd_prediction_performance_per_string,
        'avg_moving_average_ppr': avg_moving_average_ppr,
        'sd_moving_average_ppr': sd_moving_average_ppr,
        'avg_moving_average_par': avg_moving_average_par,
        'sd_moving_average_par': sd_moving_average_par
    }

    if make_plots:
        make_performance_averages_plots(avg_pred_performance_per_string=avg_prediction_performance_per_string,
                                        sd_pred_performance_per_string=sd_prediction_performance_per_string,
                                        avg_moving_average_par=avg_moving_average_par,
                                        sd_moving_average_par=sd_moving_average_par,
                                        avg_moving_average_ppr=avg_moving_average_ppr,
                                        sd_moving_average_ppr=sd_moving_average_ppr,
                                        nof_strings=nof_strings,
                                        len_inputstream=len_inputstream,
                                        save_figures=save_figures,
                                        fig_filename=fig_filename)

    return performance_metrics_average
