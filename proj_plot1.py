from __future__ import division
import os
os.chdir(r'D:\Haverford\2017-2018\Chem 362')

import pickle
from matplotlib import pyplot as plt
from matplotlib import gridspec
plt.switch_backend('Qt5Agg')

with open('plotdata.p', 'rb') as f:
    accpack, accrpack, timeused, bset, kset = pickle.load(f)
    
acc, acc1, acc2, acc12, acc3, acc4, acc34 = accpack
accr, accr1, accr2, accr12, accr3, accr4, accr34 = accrpack

"""PLOTTING"""
fig = plt.figure(figsize=(50, 20))
gs = gridspec.GridSpec(5, 2, hspace=0.4)

f0 = plt.subplot(gs[:2, :])
for numk, k in enumerate(kset):
    f0.plot(bset, acc[numk], label='$k = {k}$'.format(k=k), linestyle='-')
f0.plot(bset, accr, label='RNG results', marker='P', linestyle='--')
f0.axhline(0.62, linewidth=0.5, color='gray')
# fig.annotate('0.62', xy=(0,0.62), xytext=(0,0.62))
f0.set_title('Overall prediction accuracy')
f0.legend(loc=2)
f0.set_xlim(0, 2300)
f0.set_ylim(0, 1)

# f.xlabel('Training set/Testing set size ratio')
# f.ylabel('Percentage of correct predictions')

col1 = [acc1, acc2, acc12]
col1r = [accr1, accr2, accr12]
col1_title = ['1', '2', '1 and 2']

for i, stats in enumerate(col1):
    f = plt.subplot(gs[i + 2, 0])
    for numk, k in enumerate(kset):
        f.plot(bset, stats[numk], linestyle='-')
    f.plot(bset, col1r[i], marker='P', linestyle='--')
    f.set_title('Pred. accuracy for choice(s) %s' % col1_title[i])
    f.set_xlim(0, 2300)
    f.set_ylim(0, 1)

col2 = [acc3, acc4, acc34]
col2r = [accr3, accr4, accr34]
col2_title = ['3', '4', '3 and 4']

for i, stats in enumerate(col2):
    f = plt.subplot(gs[i + 2, 1])
    for numk, k in enumerate(kset):
        f.plot(bset, stats[numk], linestyle='-')
    f.plot(bset, col2r[i], marker='P', linestyle='--')
    f.set_title('Pred. accuracy for choice(s) %s' % col2_title[i])
    f.set_xlim(0, 2300)
    f.set_ylim(0, 1)
    
fig.suptitle('Increasing the train/test ratio improves predictions, but not enough')

plt.show()