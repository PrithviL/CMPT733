import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import StandardScaler

file=np.load("/Users/prithvi/Desktop/forest_data.npz")
X_train = file["data_training"]
Y_train = file["label_training"]

X_val = file["data_val"]
Y_val = file["label_val"]

C_values = [10.0** -3.0, 10.0** -2.0, 10.0** 10.0]
sigma = [10.0** 0.0, 10.0** 2.0, 10.0** 4.0]
gamma=[]


# Scale data
standard_scaler = StandardScaler()
X_train_norm = standard_scaler.fit_transform(X_train)
X_val_norm = standard_scaler.transform(X_val)

#X_train_norm = normalize(X_train)
#X_val_norm = normalize(X_val)

for sigma_new in sigma:
    gamma_new = 1.0* 0.5 / (sigma_new * sigma_new)
    gamma.append(gamma_new)

parameters = {'C': C_values, 'gamma': gamma}
print "***** parameters *****", parameters
svr = SVC(kernel='rbf')
score = make_scorer(accuracy_score)
CV_model = GridSearchCV(svr, parameters, cv=3, scoring=score)
CV_model.fit(X_train_norm, Y_train)

prediction = CV_model.predict(X_val_norm)
val_acc = accuracy_score(Y_val, prediction)
for i in range(3):
    print '***for fold number*** ' + str(i + 1) + ':'

    for j in C_values:
        print 'C =' + str(j) + ", accuracy = " + repr(CV_model.cv_results_)

print "The best parameter is %s " % (CV_model.best_params_)

print "The Validation accuracy is = " + str(val_acc)
