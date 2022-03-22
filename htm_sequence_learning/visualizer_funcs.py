import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(Path(__file__).parent.parent)
fig_path = os.path.join(ROOT, 'figures')


def make_performance_plots(run_idx=None,
                           pred_accuracy=None,
                           pred_performance=None,
                           pred_performance_per_string=None,
                           save_figures=True,
                           fig_filename=None):

    # ____________________________PPR and PAR PLOTS_________________________________________
    plt.figure(run_idx, figsize=(25, 7 * 3))
    plt.subplot(3, 1, 1)
    plt.plot(pred_performance, label='PPR', color='brown', lw=2)
    plt.plot(pred_accuracy, label='PAR', color='darkcyan', lw=2)
    plt.ylabel('PPR, PAR Score (in %)', fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Timestep', fontsize=25)
    plt.xticks(fontsize=25)
    plt.legend(loc='lower center', facecolor='lightgrey', fontsize=25)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    # plt.title(f'Mean prediction accuracy (in %): {np.mean([pred_[0] for pred_ in pred_acc])}')

    # _____________Frequency bar_________________
    plt.subplot(3, 1, 2)
    plt.hist([pred_performance, pred_accuracy], bins=np.arange(0, 110, 10),
             color=['brown', 'darkcyan'], label=['PPR', 'PAR'])
    plt.ylabel('Frequency of \n PPR / PAR score', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('PPR / PAR Score', fontsize=20)
    plt.xticks(np.arange(0, 110, 10), fontsize=20)
    plt.legend(loc='upper left', facecolor='lightgrey', fontsize=20)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)

    # _____________P3S PLOTS_____________________
    plt.subplot(3, 1, 3)
    plt.plot(pred_performance_per_string, label='PPR', color='darkgreen', lw=2)
    plt.ylabel('P3S Score (in %)', fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('String Index', fontsize=25)
    plt.xticks(fontsize=25)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)

    if save_figures:
        file_name = 'PAR_PPR_P3S_{}_RUNIDX{}'.format(fig_filename.replace('.npz', ''), run_idx)
        plt.savefig(fname=os.path.join(fig_path, file_name), format='svg')
        # logger.info('Figure saved in svg format at {}.svg.'.format(os.path.join(fig_path, fig_name)))

    plt.show()
    plt.close()


def make_performance_averages_plots(avg_pred_performance_per_string=None,
                                    sd_pred_performance_per_string=None,
                                    avg_moving_average_ppr=None,
                                    sd_moving_average_ppr=None,
                                    avg_moving_average_par=None,
                                    sd_moving_average_par=None,
                                    nof_strings=None,
                                    len_inputstream=None,
                                    save_figures=True,
                                    fig_filename=None):

    # PLOTTING THE AVERAGE P3S SCORE
    plt.figure(1, figsize=(28, 8 * 2))
    plt.subplot(2, 1, 1)
    plt.plot(avg_pred_performance_per_string, label='P3S', color='darkgreen', lw=2)
    plt.fill_between([i for i in range(nof_strings)],
                     avg_pred_performance_per_string - sd_pred_performance_per_string,
                     avg_pred_performance_per_string + sd_pred_performance_per_string,
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

    if save_figures:
        file_name = 'AvgP3S_MAPAR_MAPPR_{}'.format(fig_filename.replace('.npz', ''))
        plt.savefig(fname=os.path.join(fig_path, file_name), format='svg')

    plt.show()
    plt.close()


def make_htm_activations_plot():
    pass


def make_htm_predictions_plot():
    pass


def make_htm_dendrite_connection_plot():
    pass