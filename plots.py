import matplotlib.pyplot as plt

import warnings
from numpy import arange
import config as cfg
from ds_charts import set_elements
from matplotlib.font_manager import FontProperties

FONT_TEXT = FontProperties(size=6)
TEXT_MARGIN = 0.05
NR_COLUMNS: int = 3
HEIGHT: int = 4
WIDTH_PER_VARIABLE: int = 0.5



def multiple_bar_chart(xvalues: list, yvalues: dict, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '',
                       percentage: bool = False):
    ax = set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    ngroups = len(xvalues)
    nseries = len(yvalues)
    pos_group = arange(ngroups)
    series_width = pos_group[1] - pos_group[0]

    width = series_width / nseries - 0.1 * series_width
    pos_center = pos_group + series_width/2 - 0.05 * series_width
    pos_group = pos_group + width / 2
    i = 0
    legend = []
    for metric in yvalues:
        plt.bar(pos_group, yvalues[metric], width=width, edgecolor=cfg.LINE_COLOR, color=cfg.ACTIVE_COLORS[i])
        values = yvalues[metric]
        legend.append(metric)
        for k in range(len(values)):
            ax.text(pos_group[k], values[k] + TEXT_MARGIN, f'{values[k]:.2f}', ha='center', fontproperties=FONT_TEXT)
        i += 1
        pos_group = pos_group + i * width

    ax.legend(legend, fontsize='x-small', title_fontsize='small')
    plt.xticks(pos_center, xvalues)
