# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:27:52 2023

@author: bpe043
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)



#model = 'GPT-3.5'
#model = 'GPT-4'
model = 'Llama2'

""" Correctness """

correctness = pd.read_csv('{}_correctness.csv'.format(model), sep = ';')
correctness_n = correctness['count'].sum() - correctness['count'].loc[2]

fig, ax = plt.subplots(figsize = (6, 5))

sns.set_style('darkgrid')
sns.barplot(data = correctness, x = 'cat', y = 'perc', palette = 'colorblind', ax = ax)
sns.despine(offset = {'left' : 10, 'bottom' : 0})
ax.set(ylim = (0, 100))
ax.set_title('{} classification results, n = {:,.0f}'.format(model, correctness_n))
ax.set_ylabel('')
ax.set_xlabel('')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.bar_label(ax.containers[0], fmt = '%.1f%%', color = 'black')
fig.savefig('images/{}_correctness_colorblind'.format(model.replace('.', '-')), bbox_inches = 'tight', dpi = 300)


""" Archaic """

total_n = correctness['count'].sum() - correctness['count'].loc[2]

hist = pd.read_csv('{}_historical_codes.csv'.format(model), sep = ';')
hist_n = hist['count'].sum() - hist['count'].loc[2]


fig, ax = plt.subplots(figsize = (8, 6))

sns.set_style('darkgrid')
sns.barplot(data = hist, x = 'cat', y = 'perc', palette = 'colorblind', ax = ax)
sns.despine(offset = {'left' : 10, 'bottom' : 0})
ax.set(ylim = (0, 100))
ax.set_title('{} correct classification for archaic causes of death\n n = {:,.0f}, {:.2f}% of all causes of death'.format(model, hist_n, hist_n / total_n * 100))
ax.set_ylabel('')
ax.set_xlabel('')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.bar_label(ax.containers[0], fmt = '%.1f%%', color = 'black')
fig.savefig('images/{}_archaic_colorblind'.format(model.replace('.', '-')), bbox_inches = 'tight', dpi = 300)

""" Current """
non_hist = pd.read_csv('{}_non_historical_codes.csv'.format(model), sep = ';')
non_hist_n = non_hist['count'].sum() - non_hist['count'].loc[2]

fig, ax = plt.subplots(figsize = (8, 6))

sns.set_style('darkgrid')
sns.barplot(data = non_hist, x = 'cat', y = 'perc', palette = 'colorblind', ax = ax)
sns.despine(offset = {'left' : 10, 'bottom' : 0})
ax.set(ylim = (0, 100))
ax.set_title('{} correct classification for currrent causes of death\n n = {:,.0f}, {:.2f}% of all causes of death'.format(model, non_hist_n, non_hist_n / total_n * 100))
ax.set_ylabel('')
ax.set_xlabel('')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.bar_label(ax.containers[0], fmt = '%.1f%%', color = 'black')
fig.savefig('images/{}_non-historical_colorblind'.format(model.replace('.', '-')), bbox_inches = 'tight', dpi = 300)

""" Histcat """

# Correct rate
hc_correct = pd.read_csv('{}_histcat_correct_rate.csv'.format(model), sep = ';')
#hc_correct.sort_values('histcat', inplace = True)
hc_correct.sort_values('correct_rate', ascending = False, inplace = True)
hc_correct['histcat'] = hc_correct['histcat'].str.lower()


# Confusion matrix
histcat = pd.read_csv('{}_histcat_confusion_maxtrix.csv'.format(model), sep = ';')
histcat.columns = [x.lower() for x in histcat.columns]

cfm = histcat.to_numpy()

# Normalize
cfm_n = cfm.astype('float') / cfm.sum(axis = 1)[:, np.newaxis]

# Fillna, apparently, we don't have any causes of death in our dataset from the 'S' chapter
cfm_n[np.isnan(cfm_n)] = 0

# Y-axis labels needs to be added to
y_labels = histcat.columns.tolist()
new_y_labels = []
for y in y_labels:
    new_y = y + ' (' + str(hc_correct.loc[hc_correct['histcat'] == y]['correct_rate'].values[0].round(2)) + '%)'
    new_y_labels.append(new_y)

fig, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(cfm_n, annot = True, fmt = '.2f', xticklabels = histcat.columns, yticklabels = new_y_labels, cmap = 'BuPu')
ax.set_ylabel('Actual', size = 13)
ax.set_xlabel('Predicted', size = 13)

plt.show(block = False)
fig.savefig('images/{}_confusion_matrix_with_correct_perc_BlOr.png'.format(model), bbox_inches = 'tight', dpi = 300)
