import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ChartConfig:
    def __init__(self):
        self.style = 'seaborn-v0_8-whitegrid'
        self.palette = 'deep'
        self.figsize = (5, 4)
        self.dpi = 300

        self.title_fontsize = 14
        self.title_fontweight = 'bold'
        self.xlabel_fontsize = 12
        self.ylabel_fontsize = 12
        self.tick_fontsize = 10
        self.legend_fontsize = 10
        self.value_fontsize = 9

        self.bar_width = 0.35

        self.title = 'Comparison of Revenue'
        self.xlabel = 'Methods'
        self.ylabel = 'Revenue'

        self.group_labels = None

def create_bar_chart(data, config):
    plt.style.use(config.style)
    sns.set_palette(config.palette)

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize = config.figsize)

    x = np.arange(len(df['Category']))
    width = config.bar_width

    rects1 = ax.bar(x - width/2, df['Value1'], width, label=config.group_labels[0])
    rects2 = ax.bar(x + width / 2, df['Value2'], width, label=config.group_labels[1])

    # 设置x轴
    ax.set_xlabel(config.xlabel, fontsize=config.xlabel_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Category'], fontsize=config.tick_fontsize)

    ax.set_ylabel(config.ylabel, fontsize=config.ylabel_fontsize)
    ax.tick_params(axis='y', labelsize=config.tick_fontsize)

    ax.set_title(config.title, fontsize=config.title_fontsize, fontweight=config.title_fontweight)

    ax.legend(fontsize=config.legend_fontsize)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy = (rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=config.value_fontsize)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    return fig, ax

if __name__ == '__main__':
    data = {
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value1': [23, 45, 56, 78, 90],
        'Value2': [15, 30, 45, 60, 75]
    }

    config = ChartConfig()

    config.title = "Performance Comparison of Different Algorithms"
    config.xlabel = "Algorithm Types"
    config.ylabel = "Execution Time (ms)"
    config.group_labels = ['Baseline', 'Optimized']

    # 创建图表
    fig, ax = create_bar_chart(data, config)

    # 显示图表
    plt.show()

    # 保存图表
    fig.savefig('algorithm_comparison_chart.png', dpi=config.dpi, bbox_inches='tight')