from __future__ import division
import os
os.chdir(r'D:\Haverford\2017-2018\Chem 362')

import pickle, time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

"""PARAMS"""
kset = [1, 2, 3, 5]

"""DEFS"""
def Zscore_slow(df, averagetype='median', deviationtype='std'):
    # More options
    # avgtype: which number to subtract the sample with - median or mean
    # stdtype: median absolute deviation or standard deviation
    emptydf = df.copy()
    cols = list(df.columns)

    def average(series):
        if averagetype == 'median':
            return series.median()
        if averagetype == 'mean':
            return series.mean()
        
    def deviation(series):
        if deviationtype == 'mad':
            medianlist = abs(series - series.median())
            return medianlist.median()
        if deviationtype == 'std':
            return series.std(ddof=0)

    for col in cols:
        emptydf[col] = (df[col] - average(df[col]))/deviation(df[col])
    
    return emptydf

def Zscore(df):
    # Default method of the more comprehensive function defined above
    emptydf = df.copy()
    cols = list(df.columns)

    for col in cols:
        emptydf[col] = (df[col] - df[col].median())/df[col].std(ddof=0)
    
    return emptydf

# Splits arrays by conditions
def BoolSplit(arr, cond):
    
    mask = np.zeros(len(arr), dtype=bool)
    mask[cond] = True
    
    return arr[mask], arr[~mask]

"""MAIN TEXT"""
# Replace with pandas_dataframe_from_csv
df = pd.read_csv('nature17439-s2.csv')

# Remove all columns marked with XXX
# Preserve their info
# data = df[[col for col in df.columns if 'XXX' not in col]]
reality = df.iloc[:,293].copy()
rawdata = df[[col for col in df.columns if 'XXX' in col ]].copy()
reagents = rawdata.iloc[:, [1, 4, 7, 10, 13, 16]]
data = df.iloc[:,:293].drop(rawdata.columns, axis=1)

# Issue 1
# Still contains strings
# Convert data such as 'yes' and 'no' to 1s and 0s.
data = data.replace('yes', 1.0)
data = data.replace('no', 0.0)
data.fillna(value=-1.0, inplace=True)

# Issue 2
# Leave no rows with null values or strings
cols = data.columns[data.dtypes.eq('object')]
data[cols] = data[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
nullrows = data.index[data.isnull().any(1)]

data = data.dropna(axis=0, how='any')
reagents = reagents.drop(nullrows)
reality = reality.drop(nullrows)
rawdata = rawdata.drop(nullrows)

# Issue 3
# Permute the rows randomly
reality, rawdata, reagents = shuffle(reality, rawdata, reagents)

# Issue 4
# Many cols have the same entries, hence redundant to our analysis
# cols = list(df)
nunique = data.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index

datacopy = data.drop(cols_to_drop, axis=1)

print('Filtering completed')
# Issue 5
# Different columns have different scales
# Calculate Z-score of each
# Some columns might serve as better indicators than their Z-scores, e.g. pH
# Pitfalls of Z-scores: distribs. other than Gaussian, e.g. double Gaussian
Zscore_calc_start = time.clock()

data = Zscore(datacopy)

Zscore_calc_end = time.clock()
print('Z-score substitution completed')
print('Calculating the Z-score took', Zscore_calc_end - Zscore_calc_start)

def ExploratorySplit(D, R, r, testsize, exacttestsize=True):
    # D for data
    # R for reagents (so named because it has 6 columns)
    # r for reality (so named because it is 1D)
    # testsize for the desired size of testing set
    
    # exacttestsize - When the expected testing set size has been achieved, we 
    # must take away all similar reactions with it. When set to False, we set
    # all the rows taken away to be the testing set. If not, we trim down these
    # rows until its size is exactly testsize.

    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()
    
    i = 0
    while i < testsize:
        j = 0
        
        testrow = np.random.choice(R.index.values)
        testrgt = set(R.loc[testrow])
        testrgt.discard('-1')
        
        X_test = X_test.append(D.loc[[testrow]])
        Y_test = Y_test.append([r[testrow]])
        
        R = R.drop(testrow)
        j += 1
        
        for row in R.itertuples(index=True):
            idx = row[0]
            reagent = set(row[1:])
            reagent.discard('-1')
            
            # See https://stackoverflow.com/questions/16096627/selecting-a-row-of-pandas-series-Dframe-by-integer-index for difference between loc & iloc
            if reagent.issubset(testrgt):
                X_test = X_test.append(D.loc[[idx]])
                Y_test = Y_test.append([r[idx]])
                R = R.drop(idx)
                j += 1
                # print(D.loc[idx])
                # print(r[idx])
                
        i += j
    
    # Move all those are left to the training set
    trainrows = R.index.values.tolist()
    X_train = D.loc[trainrows]
    Y_train = r[trainrows]

    # Trim the rest so that the number of testing rows is testsize
    if exacttestsize == True:
        X_test = X_test.iloc[np.random.choice(i, size=testsize)]
        Y_test = Y_test.iloc[np.random.choice(i, size=testsize)]
        print('The size of the testing set is', testsize)
    else:
        print('The size of the testing set is', i)
    
    print('The size of the training set is', len(D) - i)
    
    X_test_np = np.array(X_test)
    Y_test_np = np.array(Y_test)
    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)
    
    return X_test_np, Y_test_np, X_train_np, Y_train_np

label_dist = [reality.value_counts()[i]/reality.count() for i in range(1, 5)]

package0 = ExploratorySplit(data, reagents, reality, 300, exacttestsize=False)

package = package0, label_dist

print('Train/Test split completed')

with open('cleandata.p', 'wb') as g:
    g.seek(0)
    g.truncate() # Erase everything before moving forward
    pickle.dump(package, g)

"""
domain = set(range(1,3000))
dims = range(293)
Zscore_exempted_cols = set()
uselesscols = {293}

rowdict = {i: i for i in domain}
coldict = {i: i for i in dims}

rawdata = []
reality = []

# Check if the number of rows and columns are uniform??
rowlen = set()
collen = set()

with open('nature17439-s2.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    numcol = len(header)
    for title in header:
        if 'XXX' in title:
            uselesscols.add(header.index(title))
    usefulcols = set(range(numcol)).difference(uselesscols)
    colsleft = len(usefulcols)
    allrows = [row for idx, row in enumerate(reader) if idx in domain]
    for row in allrows:
        usefulrow = [row[i] for i in usefulcols]
        actual_outcome = row[293]
        rawdata.append(usefulrow)
        reality.append(actual_outcome)


# Issue 1
# Leave no rows with null values
for i, row in enumerate(rawdata):
    if any(('?' in j) for j in row):
        for rowd in rowdict:
            if rowd > i:
                rowdict[rowd] -= 1
        del rawdata[i]
        del reality[i]
        
# Issue 2
# Still contains strings
# Convert data such as 'yes' and 'no' to 1s and 0s.
# Assign numbers to different reagents
# Convert invalid entries (-1) to something else?
rawdata0 = [[] for _ in rawdata]

for i, row in enumerate(rawdata):
    stringrows = set()
    j = 0
    while j <= colsleft - 1:
        if row[j] == None:
            rawdata0[i].append(-1.0)
        elif row[j] == 'yes':
            rawdata0[i].append(1.0)
            Zscore_exempted_cols.add(j)
            # record column number, add to exempted columns for Z-score replacement
        elif row[j] == 'no':
            rawdata0[i].append(0.0)
            Zscore_exempted_cols.add(j)
        else:
            try:
                floatij = float(row[j])
                rawdata0[i].append(floatij)
            except:
                print('String in row', i, 'and column', j)
                print(row[j])
                stringrows.add(i)
        j += 1

for k in stringrows:
    for rowd in rowdict:
        if rowd > k:
            rowdict[rowd] -= 1
    del rawdata0[k]
    k -= 1

# Issue 3
# Many cols have the same entries, hence redundant to our analysis
rawdatatrsp = list(map(list, zip(*rawdata0)))
for i in rawdatatrsp:
    if all(j == i[0] for j in i):
        for col in coldict:
            if col > rawdatatrsp.index(i):
                coldict[col] -= 1
        rawdatatrsp.remove(i)

Zscore_exempted_cols = {coldict[col] for col in Zscore_exempted_cols}

# print(rawdatatrsp)
# Produces strings, especially yes and noes

print(Zscore_exempted_cols)
print(uselesscols)

Zscore_calc_start = time.clock()

# Issue 4
# Different columns have different scales
# Calculate Z-score of each
# Some columns might serve as better indicators than their Z-scores, e.g. pH
# Pitfalls of Z-scores: distribs. other than Gaussian, e.g. double Gaussian
def Zscore(i, j):
    return (j - np.median(i))/np.std(i)

# HUGE PROBLEM - after each replacement the median is calculated with the new value
# yol bar
# SOLUTION - Use the rawdata entity again
rawdata = [[] for _ in rawdata0]

for i in rawdatatrsp:
    if rawdatatrsp.index(i) not in Zscore_exempted_cols:
        for j, item in enumerate(i):
            try:
                t = Zscore(i, item)
                rawdata[j].append(t)
            except:
                print(item, type(item), rawdatatrsp.index(i), j)
                pass

print(rawdata)

Zscore_calc_end = time.clock()
print('Calculating the Z-score took', Zscore_calc_end - Zscore_calc_start)

package = rawdata, reality, Zscore_exempted_cols, coldict, rowdict

with open('cleandata.p', 'wb') as g:
    g.seek(0)
    g.truncate() # Erase everything before moving forward
    pickle.dump(package,g)
"""

"""
Crystal size was coded with the labels 1 for no solid product, 2 for an
amorphous solid, 3 for a polycrystalline sample or 4 for single crystals with
average crystallite dimensions exceeding approximately 0.01 mm. (This size
corresponds to the general requirements for standard single-crystal X-ray diffraction
data collection.) Product purity was coded with the labels 1 for a multiphase
product or 2 for a single-phase product."""