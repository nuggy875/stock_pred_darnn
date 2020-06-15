import os
import pandas as pd

data_path = os.path.join('data/data_NAX.csv')
raw_data = pd.read_csv(data_path, index_col='Date')
targ_cols = ("KOSPI200",)                           # target Column

proc_dat = raw_data.to_numpy()
