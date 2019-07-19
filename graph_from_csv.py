#%%

import numpy as np
import csv
import json
import argparse
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

#%%
csv_file_fn = 'vgg-per_type.csv'

df = pd.read_csv(csv_file_fn)
print(df)

#%%
with open(csv_file_fn, 'r') as csv_file:
    csvr = csv.reader(csv_file)
    for row in csvr:
        
