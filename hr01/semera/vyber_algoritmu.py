import numpy;

# cd s:\semera\OneDrive\Dokumenty\Anaconda\hr_cis\priklad1 

# learning dataset
dataset = numpy.loadtxt("data/learning.csv", delimiter=",")
X = dataset[:,0:3]
Y = dataset[:,3]

# -----------------------------------------------------------------------------

from sklearn import linear_model, discriminant_analysis, neighbors, tree, naive_bayes, svm, model_selection

# modely
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', linear_model.LogisticRegression()))
models.append(('LDA', discriminant_analysis.LinearDiscriminantAnalysis()))
models.append(('KNN', neighbors.KNeighborsClassifier()))
models.append(('DT', tree.DecisionTreeClassifier()))
models.append(('NB', naive_bayes.GaussianNB()))
#models.append(('SVM', svm.SVC()))

# -----------------------------------------------------------------------------

#kalkulace
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    print("model: %s, acc: %f, std: %f" % (name, cv_results.mean()*100, cv_results.std()))