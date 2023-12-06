# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:37:29 2023

@author: bpe043
"""

import pandas as pd
import numpy as np
from pyjarowinkler.distance import get_jaro_distance as jaro
import time


"""
    Trying a string similarity script where we look for the cause of death from the infant dataset in the SHiP dictionary list of causes of death.
    If we find a match, we will grab the ICD10 code from that row
"""


# Compares the Cause of Death with potential causes of death from the dictionary. 
# Finds the one with the highest score and returns the ICD10 score that belongs to that row
def compare(cod, d, dict_col_name):
    
    if cod == '' or cod is None or type(cod) is float:
        return 'nan', 0, 0
    
    scores = {}
    
    d_causes = d[dict_col_name].dropna()
        
    for cause in d_causes:
        jaro_score = jaro(cod, cause)
        scores[cause] = jaro_score
        
    max_score_string = max(scores, key = scores.get)
    max_score_value = scores[max_score_string]
    
    index = d_causes[d_causes == max_score_string].index[0]
    code = d.loc[index]['icd10']
    
    return max_score_string, max_score_value, code


def jaro_lookup(dat, ship_dict, frame_col_name, dict_col_name):
    
    jaro_frame = pd.DataFrame(index = dat.index)
    jaro_frame[['cod', 'icd10_real', 'jaro_string', 'jaro_score', 'jaro_icd10']] = None
    
    dat = dat.reset_index(drop = True)

    start = time.time()
    for index, row in dat.iterrows():
        
        jaro_frame.at[index, 'cod'] = row[frame_col_name]
        jaro_frame.at[index, 'icd10_real'] = row['icd10']
        
        jaro_values = compare(row['cause of death'], ship_dict, dict_col_name)
        
        jaro_frame.at[index, 'jaro_string'] = jaro_values[0]
        jaro_frame.at[index, 'jaro_score'] = jaro_values[1]
        jaro_frame.at[index, 'jaro_icd10'] = jaro_values[2]
        
        perc_done = (index / len(dat)) * 100
        print('{:.2f}% done finding jaro values.'.format(perc_done))
        
    end = time.time()
    print("Assigning Jarowinkler scores to all strings took {} seconds".format(end-start))
    return jaro_frame


def direct_compare(frame, ship_frame, frame_col_name, dict_col_name):
    
    # Iterate over unique_cods, if we find an exact match in the dictionary, we grab the ICD10 code, and we update all rows in dat with that cod to have that ICD10 code
    unmatched = []
    frame['icd10_from_dict'] = np.NaN
    for u in unique_cods:
        if u in ship_frame[dict_col_name].values:
            r = ship_frame[ship_frame[dict_col_name] == u].reset_index(drop = True)
            icd10 = r['icd10'].loc[0]
            frame.loc[frame[frame_col_name] == u, 'icd10_from_dict'] = icd10
        else:
            unmatched.append(u)
            
    return  frame, unmatched

# Load in the ship masterlist for the ground truth strings
# NOTE! If you wish to enquire for a copy of the ship master list, contact the corresponding author.
ship = pd.read_excel("ship_masterlist.xlsx", sheet_name = 'masterlist')
ship.columns = [x.lower() for x in ship.columns]
ship.rename(columns = {'icd10_2level_description_english' : 'cod', 'icd10h_oct2020' : 'icd10h'}, inplace = True)
ship = ship [['cod', 'icd10', 'icd10h_description', 'icd10h', 'histcat', 'infantcat', 'typeflag']]
ship['cod'] = ship['cod'].str.lower()
ship = ship.drop_duplicates('cod')

# Load in the dataset
dat = pd.read_csv('<fileName.csv>')

# Keep the relevant columns for your project, and drop any invalid rows
dat = dat[dat['cod'].notna()].reset_index(drop = True)
dat = dat[['cod', 'cod1', 'icd10']]
dat.rename(columns = {'cod' : 'cause of death'}, inplace = True)
    

# First we can look for exact matches, and get them out of the way
unique_cods = dat['cause of death'].drop_duplicates().tolist()


# Doing a direct comparison between the cause of death string from the dataset and the cause of death column from the ship master list
frame_col_name = 'cause of death'
dict_col_name = 'cod'
dat, unmatched = direct_compare(dat, ship, frame_col_name, dict_col_name)
exact_matches_uncleaned = dat[~dat['icd10_from_dict'].isna()]     


dat_missing = dat[dat['icd10_from_dict'].isna()]

# Time to analyze the results
jaro_frame_unclearned = jaro_lookup(dat_missing, ship, 'cause of death', dict_col_name)
jaro_right = jaro_frame_unclearned[jaro_frame_unclearned['icd10_real'] == jaro_frame_unclearned['jaro_icd10']]       # 2363 rows out of 8286, or 28.52%      Combined with the exact matches: 2363 + 1396 = 3759 out of 9682, or 38.82% (892 + 26 unique codes = 918 out of 3006, or 30.54%)
jaro_wrong = jaro_frame_unclearned[jaro_frame_unclearned['icd10_real'] != jaro_frame_unclearned['jaro_icd10']]       # 5923 rows out of 8286, or 71.48%
