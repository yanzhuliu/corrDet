import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv
import os
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def read_csv(file):
    csvfile = open(file, 'r')
    pp = os.path.abspath(file)
    rows = csv.reader(csvfile, delimiter=',')
    methods = []
    next(rows, None)   # skip the head row
    for row in rows:
        mi = []
        mi.append(float(row[1]))
        mi.append(float(row[2]))
        mi.append(float(row[3]))
        mi.append(float(row[4]))
        methods.append(mi)
    return methods

# for i in range(1):
#     # set width of bar
#     barWidth = 0.06
#     fig = plt.figure()
#
#     methods = read_csv('severity_' + str(i) + '.csv')
#
#     # Set position of bar on X axis
#     br1 = np.arange(1)
#     br2 = [x + barWidth for x in br1]
#     br3 = [x + barWidth for x in br2]
#     br4 = [x + barWidth for x in br3]
#     br5 = [x + barWidth for x in br4]
#     br6 = [x + barWidth for x in br5]
#     br7 = [x + barWidth for x in br6]
#     br8 = [x + barWidth for x in br7]
#  #   br9 = [x + barWidth for x in br8]
#
#     # Make the plot
#     plt.bar(br1, methods[0], width=barWidth,
#         edgecolor='grey', label='Pretrained')
#     plt.bar(br2, methods[1], width=barWidth,
#             edgecolor='grey', label='RobustDet_cfr')
#     plt.bar(br3, methods[2], width=barWidth,
#         edgecolor='grey', label='ASAM')
#     plt.bar(br4, methods[3], width=barWidth,
#         edgecolor='grey', label='Augmix')
#     plt.bar(br5, methods[4], width=barWidth,
#         edgecolor='grey', label='Stylized')
#     plt.bar(br6, methods[5], width=barWidth,
#             edgecolor='grey', label='RobustDet')
#     plt.bar(br7, methods[6], width=barWidth,
#             edgecolor='grey', label='SAM')
#     plt.bar(br8, methods[7], width=barWidth,
#             edgecolor='grey', label='Ours')
#   #  plt.bar(br9, methods[8], width=barWidth,
#   #          edgecolor='grey', label='Ours')
#
#     # Adding Xticks
#     plt.ylabel('mPC', fontsize=20)
#     plt.xticks([])
#     plt.yticks(fontsize=12)
#     plt.ylim(0.0, 1.2)
#
#     plt.legend(fontsize=10, ncol=2)
#     plt.show()

levels = []
for i in range(1,6):

    methods = read_csv('severity_' + str(i) + '.csv')
    levels.append(methods)  # 5* 8*4
levels = np.array(levels)

for i in range(len(levels[0][0])): # corruption types
    # set width of bar
    barWidth = 0.09
    fig = plt.figure()
    # Set position of bar on X axis
    methods = levels[:,:,i]  # level*method
    br1 = np.arange(len(methods[0]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]

    # Make the plot
    plt.bar(br1, methods[0], width=barWidth,
        edgecolor='grey', label='Severity 1')
    plt.bar(br2, methods[1], width=barWidth,
            edgecolor='grey', label='Severity 2')
    plt.bar(br3, methods[2], width=barWidth,
        edgecolor='grey', label='Severity 3')
    plt.bar(br4, methods[3], width=barWidth,
        edgecolor='grey', label='Severity 4')
    plt.bar(br5, methods[4], width=barWidth,
        edgecolor='grey', label='Severity 5')

    # Adding Xticks
    plt.ylabel('mPC', fontsize=20)
    plt.xticks([r + barWidth for r in range(len(methods[:][0]))],
           ['Pretrained', 'Stylized','Augmix','SAM','ASAM','RobustDet','RobustDet_cfr','Ours'], fontsize=12,rotation = 35)
    plt.yticks(fontsize=12)
    plt.ylim(0.0, 1.0)

    plt.legend(fontsize=10, ncol=2)
    plt.show()

for i in range(1,6):
    # set width of bar
    barWidth = 0.09
    fig = plt.figure()

    methods = read_csv('severity_' + str(i) + '.csv')

    # Set position of bar on X axis
    br1 = np.arange(len(methods[0]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]
    br7 = [x + barWidth for x in br6]
    br8 = [x + barWidth for x in br7]
 #   br9 = [x + barWidth for x in br8]

    # Make the plot
    plt.bar(br1, methods[0], width=barWidth,
        edgecolor='grey', label='Pretrained')
    plt.bar(br2, methods[1], width=barWidth,
            edgecolor='grey', label='RobustDet_cfr')
    plt.bar(br3, methods[2], width=barWidth,
        edgecolor='grey', label='ASAM')
    plt.bar(br4, methods[3], width=barWidth,
        edgecolor='grey', label='Augmix')
    plt.bar(br5, methods[4], width=barWidth,
        edgecolor='grey', label='Stylized')
    plt.bar(br6, methods[5], width=barWidth,
            edgecolor='grey', label='RobustDet')
    plt.bar(br7, methods[6], width=barWidth,
            edgecolor='grey', label='SAM')
    plt.bar(br8, methods[7], width=barWidth,
            edgecolor='grey', label='Ours')
  #  plt.bar(br9, methods[8], width=barWidth,
  #          edgecolor='grey', label='Ours')

    # Adding Xticks
    plt.ylabel('mPC', fontsize=20)
    plt.xticks([r + barWidth for r in range(len(methods[0]))],
           ['noise', 'blur', 'weather', 'digital'], fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylim(0.0, 1.0)

    plt.legend(fontsize=10, ncol=2)
    plt.show()