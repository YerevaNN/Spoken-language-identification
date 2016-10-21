"""split data into training and validation sets with pandas/scikit-learn"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Read csv and drop file extension
df = pd.read_csv('trainingData.csv', names=['filename','lang'],
                 converters={'filename': lambda f: f[:-4]}, header=0)

#LabelEncoding the languages (LabelEncoder sorts values internally)
df['lang'] = LabelEncoder().fit_transform(df['lang'])

#Get first 306 rows for each lang
first_N_rows_per_lang_idx = df.groupby('lang').head(306).index
train = df.iloc[first_N_rows_per_lang_idx]
val   = df.iloc[~df.index.isin(first_N_rows_per_lang_idx)]

#Write to csv
to_csv_args = dict(sep=',', index=False, header=False)
train.to_csv('trainEqual_pandas.csv',**to_csv_args )
val.to_csv('valEqaul_pandas.csv', **to_csv_args)
