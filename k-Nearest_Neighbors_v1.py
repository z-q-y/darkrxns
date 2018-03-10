from __future__ import division
import os
os.chdir(r'D:\Haverford\2017-2018\Chem 362')

import pickle, time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Checks if the dimensions of involved arrays are in order
# Makes it easier to spot mistakes
def dims(arg):
    print(np.shape(arg))
    
kset = [1, 2, 3, 5]
with open('cleandata.p', 'rb') as f:
    rawdata0, reality0, Zscore_exempted_cols, coldict, rowdict = pickle.load(f)

# PARAMETERS
bmin = 500
bmax = 3000
bgap = 25
bset = range(0, bmax - bmin + bgap, bgap)
test_size = 0.1

traintestratio = []
timeused = [[] for _ in kset]

# DON'T USE acc, acc3, acc4, acc34 = [[[] for _ in kset]] * 4
# This is defining a single variable in many names
acc, acc1, acc2, acc12, acc3, acc4, acc34 = ([[] for _ in kset] for _ in range(7))

# Control set using randomness
accr, accr1, accr2, accr12, accr3, accr4, accr34 = ([] for _ in range(7))

reality0 = [int(i) for i in reality0]

rawdata = np.array(rawdata0)
reality = np.array([[n] for n in reality0])
label_dist = [reality0.count(i)/len(reality0) for i in range(1, 5)]

X_train, X_test, Y_train, Y_test = train_test_split(rawdata, reality,
                                                    test_size=test_size,
                                                    random_state=42)

Y0_test = np.concatenate(Y_test[:])  # 1D version of Y_test, used for zipping

# FOR SUBSET ACCURACY QUANTIFICATION USE
for b in bset:
    
    Xb_train = X_train[:bmin+b, :]
    Yb_train = Y_train[:bmin+b, :]

    print("The size of the training set is", bmin + b)
    
    traintotest = (bmin + b)/len(Y_test)
    traintestratio.append(traintotest)
    print("The ratio of training set to testing set is", traintotest)
    
    for k, n_neighbors in enumerate(kset):
        kstart = time.clock()
        
        knn = KNeighborsClassifier(n_neighbors)    
        knn.fit(Xb_train, Yb_train)
        pred = knn.predict(X_test)
        
        # Prediction choices?
        _, predcounts = np.unique(pred, return_counts=True)
        if len(predcounts) != 4:
            mask = np.isin(list(range(1, 5)), pred)
            for i in np.where(mask == False):
                predcounts = np.insert(predcounts, i, 1)
                # change to 0 in future, involves amending below
        
        zipcheck = np.column_stack((Y0_test, pred))
        
        right_choices, right_choices_1, right_choices_2, right_choices_1_2, right_choices_3, right_choices_4, right_choices_3_4= [0] * 7
        
        for i, (a, b) in enumerate(zipcheck):
            if a == b:
                right_choices += 1
            if a == b == 1:
                right_choices_1 += 1
            if a == b == 2:
                right_choices_2 += 1
            if a == b == 3:
                right_choices_3 += 1
            if a == b == 4:
                right_choices_4 += 1
            if a in (1, 2) and b in (1, 2):
                right_choices_1_2 += 1
            if a in (3, 4) and b in (3, 4):
                right_choices_3_4 += 1
                
        accuracy = right_choices/len(pred)
        print("The predictions are", accuracy * 100, "percent correct")
        
        accuracy_1 = right_choices_1/predcounts[0]
        accuracy_2 = right_choices_2/predcounts[1]
        accuracy_1_2 = right_choices_1_2/(predcounts[0] + predcounts[1])
        accuracy_3 = right_choices_3/predcounts[2]
        accuracy_4 = right_choices_4/predcounts[3]
        accuracy_3_4 = right_choices_3_4/(predcounts[2] + predcounts[3])
                    
        acc[k].append(accuracy)
        acc1[k].append(accuracy_1)
        acc2[k].append(accuracy_2)
        acc12[k].append(accuracy_1_2)
        acc3[k].append(accuracy_3)
        acc4[k].append(accuracy_4)
        acc34[k].append(accuracy_3_4)
        
        kend = time.clock()
        timeused[k].append(kend - kstart)
        print("The operation took", kend - kstart, "to complete")
        
    # Baselines           
    # Weighted randomly generated results are presented as a control group
    pred_random = np.random.choice(np.array(range(1, 5)), int(len(Y_test)), p=label_dist)
    _, predcounts_random = np.unique(pred_random, return_counts=True)
    if len(predcounts_random) != 4:
            print('Divide by zero scenario ahead')
            raise ValueError('For a certain outcome, RNG generates zero appearances')
    
    zipcheck_random = np.column_stack((Y_test, pred_random))
            
    random_choices, random_choices_1, random_choices_2, random_choices_1_2, random_choices_3, random_choices_4, random_choices_3_4= [0] * 7
    
    for i, (a, b) in enumerate(zipcheck_random):
        if a == b:
            random_choices += 1
        if a == b == 1:
            random_choices_1 += 1
        if a == b == 2:
            random_choices_2 += 1
        if a == b == 3:
            random_choices_3 += 1
        if a == b == 4:
            random_choices_4 += 1
        if a in (1, 2) and b in (1, 2):
            random_choices_1_2 += 1
        if a in (3, 4) and b in (3, 4):
            random_choices_3_4 += 1
            
    accuracy_r = random_choices/len(pred_random)
    print("The predictions are", accuracy * 100, "percent correct")
    
    accuracy_r1 = random_choices_1/predcounts_random[0]
    accuracy_r2 = random_choices_2/predcounts_random[1]
    accuracy_r1_r2 = random_choices_1_2/(predcounts_random[0] + predcounts_random[1])
    accuracy_r3 = random_choices_3/predcounts_random[2]
    accuracy_r4 = random_choices_4/predcounts_random[3]
    accuracy_r3_r4 = random_choices_3_4/(predcounts_random[2] + predcounts_random[3])
                
    accr.append(accuracy_r)
    accr1.append(accuracy_r1)
    accr2.append(accuracy_r2)
    accr12.append(accuracy_r1_r2)
    accr3.append(accuracy_r3)
    accr4.append(accuracy_r4)
    accr34.append(accuracy_r3_r4)
    
    # Code used for single random benchmark
    """
    pred_random = np.reshape(pred_random, (-1, 1))
    accuracy_random = accuracy_score(Y_test, pred_random)
    acc_random.append(accuracy_random)
    """
    
    b += bgap

accpack = acc, acc1, acc2, acc12, acc3, acc4, acc34
accrpack = accr, accr1, accr2, accr12, accr3, accr4, accr34
plotuse = accpack, accrpack, timeused, bset, kset

with open('plotdata.p', 'wb') as g:
    g.seek(0)
    g.truncate() # Erase everything before moving forward
    pickle.dump(plotuse,g)