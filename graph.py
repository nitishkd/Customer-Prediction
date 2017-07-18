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

#import pickle
#s = pickle.dumps(clf)
##clf2 = pickle.loads(s)

##print(clf2.predict([[22,3,4]]))

X = FD[:, 1:3]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = Y

h = .02  # step size in the mesh
C = 1.0  # SVM regularization parameter
##svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = FD[:, 2].min() - 1, FD[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = [##'SVC with linear kernel',
          ##'LinearSVC (linear kernel)',
          'SVC with RBF kernel'##
          ##'SVC with polynomial (degree 3) kernel'
          ]


#for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
for i, clf in enumerate((rbf_svc,)):
    plt.plot(2,2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('EXP')
    plt.ylabel('FREQ')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()


