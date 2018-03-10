from __future__ import division
import os
os.chdir(r'D:\Haverford\2017-2018\Chem 362')

import csv, pickle, time
import numpy as np

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
        
# Issue 1
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

# Issue 2
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

# Issue 3
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

# Issue 4
# Any advantage in defining data as tuples?

Zscore_calc_end = time.clock()
print('Calculating the Z-score took', Zscore_calc_end - Zscore_calc_start)

package = rawdata, reality, Zscore_exempted_cols, coldict, rowdict

with open('cleandata.p', 'wb') as g:
    g.seek(0)
    g.truncate() # Erase everything before moving forward
    pickle.dump(package,g)

"""Crystal size was coded with the labels 1 for no solid product, 2 for an
amorphous solid, 3 for a polycrystalline sample or 4 for single crystals with
average crystallite dimensions exceeding approximately 0.01 mm. (This size
corresponds to the general requirements for standard single-crystal X-ray diffraction
data collection.) Product purity was coded with the labels 1 for a multiphase
product or 2 for a single-phase product."""