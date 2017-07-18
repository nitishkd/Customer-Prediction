import numpy as np;
import matplotlib.pyplot as plt
from sklearn import svm

dset_file = open("extradata.txt");
##print(dset_file.readline())
row = 0;
col = 0;
dset = []
for info in dset_file.readlines(): 
    col = 0;
    s = info.strip()
    sa = s.split("(")
    sb = sa[1].split(")")
    k = sb[0].split(",")
    lst = []
    for ss in k:
        val = int(ss)
        lst.append(val)
    dset.append(lst)

FD = np.array(dset)  
X = FD[:, :3]
Y = FD[:, 3]

clf = svm.SVC()
clf.fit(X,Y)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)

res = clf2.predict([[22,5,4]])
result = str(res[0])
print result