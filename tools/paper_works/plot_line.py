import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv

def read_csv(file):
    csvfile = open(file, 'r')
    rows = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    next(rows, None)   # skip the head row
    for row in rows:
        y.append(float(row[2]))
        x.append(int(row[1]))
    return x, y

#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'NSimSun, Times New Roman'

fig = plt.figure()
#ax = fig.add_subplot(111)
# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
# axes = axes.flatten()
#
# for ax in axes:
#     ax.set_ylabel(r'$\ln\left(\frac{x_a-x_b}{x_a-x_c}\right)$')
#     ax.set_xlabel(r'$\ln\left(\frac{x_a-x_d}{x_a-x_e}\right)$')

x1, y1 = read_csv('voc_sam_combined/loss_bbox.csv')
plt.plot(x1, y1, color='red', label='loss_bbox')

x2, y2 = read_csv('voc_sam_combined/loss_cls.csv')
plt.plot(x2, y2, 'g', label = 'loss_cls')

x1, y1 = read_csv('voc_sam_clean/loss_bbox.csv')
plt.plot(x1, y1, color='red', label='loss_bbox_clean')

x2, y2 = read_csv('voc_sam_clean/loss_cls.csv')
plt.plot(x2, y2, 'g', label = 'loss_cls_clean')

# x3, y3 = read_csv('voc_sam_combined/loss_rpn_bbox.csv')
# plt.plot(x3, y3, color='blue', label = 'loss_rpn_bbox')
#
# x4, y4 = read_csv('voc_sam_combined/loss_rpn_cls.csv')
# plt.plot(x4, y4, color='black', label = 'loss_rpn_cls')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

#plt.ylim(0, 0.6)
#plt.xlim(1, 622)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()