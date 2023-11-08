import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv
import numpy as np

def read_csv(file):
    csvfile = open(file, 'r')
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

for i in range(5):
    # set width of bar
    barWidth = 0.09
    fig = plt.figure()

    methods = read_csv('severity_' + str(i+1) + '.csv')

    # Set position of bar on X axis
    br1 = np.arange(len(methods[0]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]
    br7 = [x + barWidth for x in br6]
    br8 = [x + barWidth for x in br7]
    br9 = [x + barWidth for x in br8]

    # Make the plot
    plt.bar(br1, methods[0], width=barWidth,
        edgecolor='grey', label='pre-trained')
    plt.bar(br2, methods[1], width=barWidth,
        edgecolor='grey', label='stylized')
    plt.bar(br3, methods[2], width=barWidth,
        edgecolor='grey', label='augmix')
    plt.bar(br4, methods[3], width=barWidth,
            edgecolor='grey', label='aug-stylized')
    plt.bar(br5, methods[4], width=barWidth,
        edgecolor='grey', label='SAM')
    plt.bar(br6, methods[5], width=barWidth,
        edgecolor='grey', label='ASAM')
    plt.bar(br7, methods[6], width=barWidth,
        edgecolor='grey', label='AWP')
    plt.bar(br8, methods[7], width=barWidth,
            edgecolor='grey', label='RWP')
    plt.bar(br9, methods[8], width=barWidth,
            edgecolor='grey', label='Ours')

    # Adding Xticks
    plt.ylabel('mPC', fontsize=20)
    plt.xticks([r + barWidth for r in range(len(methods[0]))],
           ['noise', 'blur', 'weather', 'digital'], fontsize=15)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=10, ncol=2)
    plt.show()