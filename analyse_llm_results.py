# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:46:01 2023

@author: bpe043
"""

import pandas as pd
import numpy as np
import functools as ft
from sklearn.metrics import confusion_matrix

""" Define the model who's results you are analying. For instance 'GPT-3.5' """
model = '<modelName>'

""" Load in and prepare files """

# Read in the gold standard dataset you use to compare with the LLM results
dat = pd.read_csv('<fileName>.csv')

# Keep the relevant columns and remove any invalid data.
# (I've left our original column names in the script, as they are referenced a lot later on, but replace them with your own column names)
dat = dat[['deathid', 'cod', 'cod1', 'cod1code', 'icd10', 'chapter', 'histcat']]
dat.rename(columns = {'cod1code' : 'icd10h'}, inplace = True)
dat = dat[dat['cod'].notna()].reset_index(drop = True)


# The result file you get from the LLM, just replace the filename
res = pd.read_csv('{}_output.csv'.format(model))
res = res[res['cod'].notna()].reset_index(drop = True)

# The important column in the result file is primarily what I have called "gpt_icd10", this is the LLM prediction for what the ICD10 code should be. 
# (Once again, replace these variables with whatever columns are relevant for your own project)
res = res[['cod', 'histcat', 'infantcat', 'icd10', 'gpt_icd10']]

# The 'res' frame might need to be cleaned, if for example the LLM returned multiple codes
# But since we are really only looking for the FIRST ICD10 code, that's the one we separate out
res['llm_icd10_1'] = res['gpt_icd10'].str.split(',').str[0]

# Since we are primarily interested in the first ICD10 code (since we assume we can do everything with the other ICD10 columns later on at some point)
# We add that column to the 'dat' frame
dat['llm_icd10'] = res['llm_icd10_1']

# And now we can do all sorts of analysis on it

""" How many did the LLM get correct? """

# Like find out how many CoDs (Causes of death) could not get an ICD10 code from the LLM
nans = dat[dat['llm_icd10'].isna()]

# See how many of the codes the LLM got Completely right
dat['full_match'] = False
dat_full_match_index = dat[dat['icd10'] == dat['llm_icd10']]
dat['full_match'].loc[dat_full_match_index.index] = True 

# And how many it got Partially correct, meaning the first 3 characters before the dot.
dat['partial_match'] = False
dat_partial_match_index = dat[dat['icd10'].str[0:2] == dat['llm_icd10'].str[0:2]]
dat['partial_match'].loc[dat_partial_match_index.index] = True

# Then we actually grab the rows that belong to Full_match and Partial_match
full_match_check = dat[(dat['full_match'] == True) & (dat['partial_match'] == True)]
partial_match_check = dat[(dat['full_match'] == False) & (dat['partial_match'] == True)]

# And how many were just completely wrong
incorrect = dat[(dat['full_match'] == False) & (dat['partial_match'] == False)]  

# How many of the incorrect codes used the "error code" we defined for the model? "Æ99.9"
error_codes = incorrect[incorrect['llm_icd10'] == 'Æ99.9']

# How many of the incorrect codes were not "errors" but just wrong?
wrong_codes = incorrect[incorrect['llm_icd10'] != 'Æ99.9']

correctness_out = pd.DataFrame(columns = ['cat', 'count', 'perc'])
correctness_out.loc[0] = ['Full match', len(full_match_check), (len(full_match_check) / len(dat)) * 100]
correctness_out.loc[1] = ['Partial match', len(partial_match_check), (len(partial_match_check) / len(dat)) * 100]
correctness_out.loc[2] = ['Correct (full+partial)', len(full_match_check) + len(partial_match_check), ((len(full_match_check) + len(partial_match_check) )/ len(dat)) * 100]
correctness_out.loc[3] = ['Errors', len(incorrect), (len(incorrect) / len(dat)) * 100]
correctness_out.to_csv('{}_correctness.csv'.format(model), index = False)


""" Does it matter what the ICD10h code ends in? / Historical codes"""

# Historical code does NOT end in a 0, but everything else. So here we grab the codes that ends in anything other than 0
hist = dat[~dat['icd10h'].str.endswith('0')]                                                        
hist_full_match = hist[hist['full_match'] == True]
hist_partial_match = hist[(hist['full_match'] == False) & (hist['partial_match'] == True)]
hist_errors = hist[(hist['full_match'] == False) & (hist['partial_match'] == False)]

hist_out = pd.DataFrame(columns = ['cat', 'count', 'perc'])
hist_out.loc[0] = ['Full match', len(hist_full_match), (len(hist_full_match) / len(hist)) * 100]
hist_out.loc[1] = ['Partial match', len(hist_partial_match), (len(hist_partial_match) / len(hist)) * 100]
hist_out.loc[2] = ['Correct (full + partial)', len(hist_full_match) + len(hist_partial_match), ((len(hist_full_match) + len(hist_partial_match) )/ len(hist)) * 100]
hist_out.loc[3] = ['Errors', len(hist_errors), (len(hist_errors) / len(hist)) * 100]
hist_out.to_csv('{}_historical_codes.csv'.format(model), index = False)

# And then, by necessity, everything else in the dat frame is a Non-historical code, which will end in a 0
non_hist = dat.loc[~dat.index.isin(hist.index)]
non_hist_full_match = non_hist[non_hist['full_match'] == True]        
non_hist_partial_match = non_hist[(non_hist['full_match'] == False) & (non_hist['partial_match'] == True)]
non_hist_errors = non_hist[(non_hist['full_match'] == False) & (non_hist['partial_match'] == False)]

non_hist_out = pd.DataFrame(columns = ['cat', 'count', 'perc'])
non_hist_out.loc[0] = ['Full match', len(non_hist_full_match), (len(non_hist_full_match) / len(non_hist)) * 100]
non_hist_out.loc[1] = ['Partial match', len(non_hist_partial_match), (len(non_hist_partial_match) / len(non_hist)) * 100]
non_hist_out.loc[2] = ['Correct (full + partial)', len(non_hist_full_match) + len(non_hist_partial_match), ((len(non_hist_full_match) + len(non_hist_partial_match) )/ len(non_hist)) * 100]
non_hist_out.loc[3] = ['Errors', len(non_hist_errors), (len(non_hist_errors) / len(non_hist)) * 100]
non_hist_out.to_csv('{}_non_historical_codes.csv'.format(model), index = False)



""" Number of words """
dat['total_words'] = dat['cod'].str.count(' ') + 1
words_total_overview = dat['total_words'].value_counts().reset_index()
words_total_overview_perc = dat['total_words'].value_counts(normalize = True).reset_index()

words_full_match = dat[(dat['full_match'] == True) & (dat['partial_match'] == True)]
words_partial_match = dat[(dat['full_match'] == False) & (dat['partial_match'] == True)]
words_errors = dat[(dat['full_match'] == False) & (dat['partial_match'] == False)]

words_full_match_abs = words_full_match['total_words'].value_counts().sort_index().reset_index().rename(columns = {'count' : 'count_full_match'})
words_partial_match_abs = words_partial_match['total_words'].value_counts().sort_index().reset_index().rename(columns = {'count' : 'count_partial_match'})
words_errors_abs = words_errors['total_words'].value_counts().sort_index().reset_index().rename(columns = {'count' : 'count_errors'})

words = [words_total_overview, words_full_match_abs, words_partial_match_abs, words_errors_abs]
words = ft.reduce(lambda left, right: pd.merge(left, right, on = 'total_words', how = 'left'), words)
words = words.sort_values('total_words')

words = words.fillna(0)
words['match_rate'] = (words['count_full_match'] + words['count_partial_match']) / words['count'] * 100
words['error_rate'] = words['count_errors'] / words['count'] * 100

words_median_error = words.error_rate.median()
words_avg_error = words.error_rate.sum() / len(words)

words_median_error_short_words = words[words['total_words'] < 6].error_rate.median()
words_median_error_medium_words = words[words['total_words'] > 5 & (words['total_words'] <= 10)].error_rate.median()
words_median_error_long_words = words[words['total_words'] > 10].error_rate.median()

words.to_csv('{}_word_length.csv'.format(model), sep = ';', index = False)


""" Histcat """            

""" 
    What we hope to find:
    Are the errors made by the model Big errors, meaning that they were given the code for an entirely different category of causes of death?
    Or were they given the wrong code, but still a code within the same category?
"""

""" Histcats that were classified correctly """
hc_tot_overview = dat.histcat.value_counts().reset_index()
hc_correct = pd.concat([full_match_check, partial_match_check])
hc_correct = hc_correct.histcat.value_counts().reset_index()

hc_correct_overview = pd.merge(how = 'left', left = hc_tot_overview, right = hc_correct, left_on = hc_tot_overview['histcat'], right_on = hc_correct['histcat'], suffixes=(['_tot', '_correct']))
hc_correct_overview = hc_correct_overview.drop(['histcat_tot', 'histcat_correct'], axis = 1)
hc_correct_overview.rename(columns = {'key_0' : 'histcat'}, inplace = True)
hc_correct_overview['correct_rate'] = hc_correct_overview['count_correct'] / hc_correct_overview['count_tot'] * 100

hc_correct_overview.to_csv('{}_histcat_correct_rate.csv'.format(model), sep = ';', index = False)

# Order by correctness_rate
#hc_correct_overview.sort_values('correct_rate', ascending = False, inplace = True)

###
	NOTE! 
	In order to run the code below, you would need access to the master list of ICD-10h codes assigned to causes of death,
	which were created by the SHiP+ network. If you wish to ask for a copy of this list, contact the corresponding author of the article. 
###

ship = pd.read_excel("ship_masterlist.xlsx", sheet_name = 'masterlist')
ship.columns = [x.lower() for x in ship.columns]

hc_temps = pd.DataFrame(columns = dat.columns)
unique_llm_icd10_codes = incorrect['llm_icd10'].drop_duplicates().tolist()
unique_llm_icd10_codes = [x for x in unique_llm_icd10_codes if x is not np.NaN]
not_found = []
for u in unique_llm_icd10_codes:

    temp = ship[ship['icd10'] == u]
    if len(temp) != 0:
        hc_temps = pd.concat([hc_temps, temp])
    else:
        not_found.append(u)

gt = hc_temps[['cod', 'icd10', 'histcat', 'llm_icd10']]
gt_unique = gt.drop_duplicates(subset = ['icd10', 'histcat'])

# Found two icd10 codes with multiple histcats, R99 which had [ill defined, stated to be 'unknown', not a cause of death, no cause given/blank] 
# and K63.9 which had the histcats [Digestive, and Diarrhoea] --> K63.9 is classified in the ICD-10 as "Other diseases of intestine -- Disease of intestine, unspecified" So, I will decide to keep only the histcat for "Digestive"
gt_unique['histcat'].loc[gt_unique['icd10'] == 'K63.9'] = 'Digestive'

#TODO: For now, set R99's histcat to be "Ill defined"
gt_unique['histcat'].loc[gt_unique['icd10'] == 'R99'] = 'Ill defined'

gt_unique = gt_unique.drop_duplicates(subset = ['icd10', 'histcat'])
gt_unique = gt_unique[['icd10', 'histcat']]
gt_unique.rename(inplace = True, columns = {'icd10' : 'to_drop', 'histcat' : 'llm_icd10_histcat'})

# Now there are 500 unique ICD-10 codes in gt_unique. with their corresponding "correct" Histcat grabbed from the ship master list
# There are still 100 potential codes that Could be added to this dataframe, but at least some of them I know are not in the ship master list at all (Error code I created, some that are plain to see are hallucinations)
# and I doubt we would get too many other unique codes that are not already in the dataframe, even if we spend more time trying to make it work. 

hc_errors = incorrect.copy()
hc_errors = pd.merge(how = 'left', left = hc_errors, right = gt_unique, left_on = hc_errors['llm_icd10'], right_on = gt_unique['to_drop'])
hc_errors = hc_errors.drop(['full_match', 'partial_match', 'to_drop', 'key_0'], axis = 1)

# We separate out the rows which we don't have a llm_icd10_histcat for, this is due to either having a llm_icd10 code that couldn't be found in the master list (error code and hallucinations)
# or because the cause had no llm_icd10 code at all --> Incorrect formatting by the model
hc_errors_unfindable = hc_errors[hc_errors['llm_icd10_histcat'].isna()]
hc_errors = hc_errors.drop(hc_errors_unfindable.index, axis = 0)

hc_true = hc_errors['histcat'].tolist()
hc_pred = hc_errors['llm_icd10_histcat'].tolist()
labels = list(set(hc_true))
labels.sort()

# If we want to order by Correctness rate
#labels = hc_correct_overview['histcat'].tolist()

hc_conf = confusion_matrix(hc_true, hc_pred, labels = labels)
hc_conf_frame = pd.DataFrame(hc_conf)
hc_conf_frame.columns = [x for x in labels]
hc_conf_frame.to_csv('{}_histcat_confusion_maxtrix.csv'.format(model), index = False)















