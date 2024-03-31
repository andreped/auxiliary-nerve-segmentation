import matplotlib.pyplot as plt
import numpy as np

means_recall_no_augmentations = [0.73978329, 0.45072553, 0.56485409, 0.45588675, 0.23950081]
means_precision_no_augmentations = [0.83592767, 0.72035694, 0.768848,   0.713673,   0.48667899]
std_recall_no_augmentations = [0.22197877, 0.36119074, 0.3843458,  0.34753692, 0.33639544]
std_precision_no_augmentations = [0.1650925,  0.36374009, 0.32479149, 0.25966749, 0.38859466]

means_recall_all_augmentations = [0.80911386, 0.56138211, 0.71398807, 0.61713153, 0.33154735]
means_precision_all_augmentations = [0.88476378, 0.7843768,  0.82247168, 0.69379067, 0.45731968]
std_recall_all_augmentations = [0.16831188, 0.37857634, 0.32894021, 0.30623862, 0.28443283]
std_precision_all_augmentations = [0.13501367, 0.27026907, 0.2796976,  0.24998759, 0.3191905 ]


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width(), 1.02*height,
                '%.2f' % height,
                ha='center', va='bottom')

fig, ax = plt.subplots(figsize=(10,5))
width = 0.2
indexes = np.arange(5)
precision_per_object = np.ndarray((5,))
precision_std_per_object = np.ndarray((5,))
recall_per_object = np.ndarray((5,))
recall_std_per_object = np.ndarray((5,))
p1 = ax.bar(indexes, means_recall_no_augmentations, width, yerr=std_recall_no_augmentations, color='lightblue')
p2 = ax.bar(indexes + width, means_recall_all_augmentations, width, yerr=std_recall_all_augmentations, color='lightgreen')
p3 = ax.bar(indexes + width*2, means_precision_no_augmentations, width, yerr=std_precision_no_augmentations, color='blue')
p4 = ax.bar(indexes + width*3, means_precision_all_augmentations, width, yerr=std_precision_all_augmentations, color='green')
ax.set_xticks(indexes + width*3 / 2)
ax.set_xticklabels(('Blood vessel', 'MSC nerve', 'Median nerve', 'Ulnar nerve', 'Radial nerve'))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Recall (No augm.)', 'Recall (All augm.)', 'Precision (No augm.)', 'Precision (All augm.)'))
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1.1, 0.1))
autolabel(p1)
autolabel(p2)
autolabel(p3)
autolabel(p4)
#plt.savefig(join('results', experiment_name, 'final_plot.png'), bbox_inches='tight')
plt.show()