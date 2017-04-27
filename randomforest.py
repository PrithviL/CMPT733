import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier


file=np.load("/Users/prithvi/Desktop/forest_data.npz")
X_train = file["data_training"]
Y_train = file["label_training"]

X_val = file["data_val"]
Y_val = file["label_val"]

#Normalizing the feature vectors
X_train_norm = normalize(X_train)
X_val_norm = normalize(X_val)

#random forest with datapoint sampling and max features

feature_sampling = [0.2, 0.4, 0.6, 0.8 ]
data_sampling = [0.2, 0.4, 0.6, 0.8]
forest_size = np.array([10, 20, 50])

forest_size=[10,20,50]
for size in forest_size:
    for sample in data_sampling:
        model = RandomForestClassifier(n_estimators=size, max_depth=100, \
        bootstrap=True, \
        class_weight='balanced', oob_score=True, n_jobs=-1)
        model.fit(X_train_norm, Y_train, sample)
        print "oobscore", (1 - model.oob_score_)
        print "Forest size", size
        print "samples", sample

for size in forest_size:
    for sample in feature_sampling:
        model = RandomForestClassifier(n_estimators=size, max_depth=100, \
        max_features=sample, bootstrap=True, \
        class_weight='balanced', oob_score=True, n_jobs=-1)
        model.fit(X_train_norm, Y_train)
        print "oobscore", (1 - model.oob_score_)
        print "Forest size" , size
        print "samples",sample



#for f in feature_samples:
 #   rfc=RandomForestClassifier(n_estimators=f ,max_depth=100,max_features=[10,20,50],oob_score=True,class_weight='balanced')

#rfc.fit(X_train_norm,Y_train)
#print "RF",rfc.fit



