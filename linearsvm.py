import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score


file=np.load("/Users/prithvi/Desktop/forest_data.npz")
X_train = file["data_training"]
Y_train = file["label_training"]

X_val = file["data_val"]
Y_val = file["label_val"]

C_values = [0.001, 0.01, 10000000000]

# Scale data
standard_scaler = StandardScaler()
X_train_norm = standard_scaler.fit_transform(X_train)
X_val_norm = standard_scaler.transform(X_val)

#X_train_norm = normalize(X_train)
#X_val_norm = normalize(X_val)

parameters = {'C': [0.001, 0.01, 10000000000]}
svr = LinearSVC()
score = make_scorer(accuracy_score)
CV_model = GridSearchCV(svr, parameters, cv=3, scoring=score)
CV_model.fit(X_train_norm, Y_train)

for i in range(3):
    print '***** for fold *****' + str(i + 1) + ':'
    for j in C_values:
        print 'C =' + str(j) + ", accuracy = " + repr(CV_model.cv_results_)

print "The best parameter is %s " % (CV_model.best_params_)

prediction = CV_model.predict(X_val_norm)

print "The Validation accuracy is = " + repr(accuracy_score(Y_val, prediction))
