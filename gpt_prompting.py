# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:50:23 2023

@author: bpe043
"""

import openai
import pandas as pd
import re
import numpy as np

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
    )

with open('key.txt', 'r') as f:
    openai.api_key = f.read()
    
      
# This function will get ChatGPT instructional messages for both rules it has to follow, and specifications of instructions
# Added a retry decorator to add in a sleep period if we exceed our rate limit. If that happends, the script won't crash, instead it sleeps for a while and then tries again
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat_completion_instructions(prompt, model = 'gpt-4'):
    messages = [
        {'role' : 'system', 'content' : """Assistant is an intelligent chatbot designed to help the user assign clinical ICD-10 codes to causes of death.
         Instructions:
         - Only answer using standard ICD-10 codes, do not use ICD-10-CM billing codes.
         - Only return a single ICD-10 code per injury and/or disease found in the given cause of death.
         - Each ICD-10 code should not consist of more than 5 characters, the typical format looks like this: 'X01.0'
         - Your answers should be in the following format: 'Cause of death: <CAUSE OF DEATH>, ICD-10 code: <ICD-10 CODE>'
         - If you are unsure of an answer, do not try to guess. Instead, write the following reply: 'Cause of death: Unknown, ICD-10 code: Æ99.9'. 
         """},
         {'role' : 'user', 'content' : prompt + '?'}
        ]
         
    response = openai.ChatCompletion.create(model = model, messages = messages, temperature = 0.7)
    
    return response.choices[0].message['content']



# Load in your (historical death) data
dat = pd.read_csv('<fileName>.csv')


# Now, iterate over each cause of death in dat, and prompt the model for an output

# Not optimal, but iterrows was used
dat['gpt_cod'] = np.NaN
dat['gpt_icd10'] = np.NaN
raw_responses = []

for index, row in dat.iterrows():
    prompt = row['cod']
    
    if prompt is np.NaN:
        continue
    
    response = get_chat_completion_instructions(prompt, model = "gpt-4-0314")
    raw_responses.append(response)
    
# =============================================================================
#     # Regex to grab the disease/injury terms from the response
#     cods = re.findall(r'[a-zA-Z]+', response)
#     cods = [x for x in cods if len(x) > 1]
# =============================================================================
    
    # Regex to grab all codes from the response
    codes = re.findall(r'(?:[A-Z]\d+\.\d+)|(?:[A-Z]\d+)', response)
    
    # Test to find out if GPT gave the error code we defined, or if it actually didn't return any codes
    if len(codes) == 0:
        error_test = 'Æ99.9' in response
        if error_test is True:
            codes = ['Æ99.9']

    
    # Normal logic to grab cause of death output
    # First if GPT gave several causes of death as output, then if it only gave one
    line_test = response.split('\n')
    if len(line_test) > 1:
        cods = []
        for l in line_test:
            
            if len(l) == 0:
                continue
            
            cod = l.split(',')[0]
            cod = cod.split(':')
            
            if len(cod) > 1:
                cod = cod[1].strip()
            else:
                cod = 'Incorrectly formatted response'
            
            cods.append(cod)
        if len(cods) == 0:
            dat.at[index, 'gpt_cod'] = np.NaN
        else:
            dat.at[index, 'gpt_cod'] = ', '.join(cods)
    else:
        cod = response.split(',')[0]
        cod = cod.split(':')
        if len(cod) == 2:
            cod = cod[1].strip()
        else:
            cod = 'ERROR at {}'.format(index)
            
        dat.at[index, 'gpt_cod'] = cod
        
    if len(codes) == 0:
        dat.at[index, 'gpt_icd10'] = np.NaN
    else: 
        dat.at[index, 'gpt_icd10'] = ', '.join(codes)
    
    print(index)
    
    
    
    
# Print results 
dat.to_csv('<outputFileName>.csv', index = False)
