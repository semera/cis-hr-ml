# zdroje
# http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/
# http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/

#cd s:\semera\OneDrive\Dokumenty\Anaconda\hr_cis\priklad1 

import numpy;
from sklearn import tree, model_selection;

# learning dataset
dataset = numpy.loadtxt("data/learning.csv", delimiter=",")
X = dataset[:,0:3]
Y = dataset[:,3]

# test dataset, ten splitovat nemusim
preX = numpy.loadtxt("data/test1000.csv", delimiter=",") 


# -----------------------------------------------------------------------------

# decision tree clasifier
model = tree.DecisionTreeClassifier(max_depth=2);
#model = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=2, min_samples_leaf=5);
model.fit(X, Y);

# -----------------------------------------------------------------------------

# test accuracy, lepsi by to bylo mit na jinych datech, ale u tree staci..
scoring = 'accuracy';
seed = 7;
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("acc: %f, std: %f" % (cv_results.mean() * 100, cv_results.std()))


# -----------------------------------------------------------------------------

# http://www.webgraphviz.com/ 
# export do graphwiz
tree.export_graphviz(model.tree_, out_file='tree.dot',  feature_names= ['typ', 'O1', 'O2'])

# -----------------------------------------------------------------------------

# vysledna predikce a tisk, 995 bodu
predictions = model.predict(preX);
for p in predictions:
    print(int(p))
    