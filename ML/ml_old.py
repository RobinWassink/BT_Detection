from matplotlib.pyplot import xticks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from pyod.models.lof import LOF
from sklearn import metrics
import ast
import os,sys
import pickle
import traceback
import numpy as np

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def train(train_data, test_data, clsname, total_test_data):
    X = np.array(train_data)
    y = [1 for i in range(0,len(X))] 
    outliers_fraction = 0.05
    if X.ndim == 3:
        nsamples, nx, ny = X.shape
        X = X.reshape((nsamples,nx*ny))
    print(X.shape)
    print(X.ndim)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, shuffle=False)
    y_val = [1 for i in range(0,len(X_val))]

    classifiers = {
            "Robust covariance": EllipticEnvelope(contamination=outliers_fraction , support_fraction=0.5),
            "One-Class SVM": OneClassSVM(cache_size=200, gamma='scale', kernel='rbf',nu=0.05,  shrinking=True, tol=0.001,verbose=False),
            "SGD One-Class SVM": SGDOneClassSVM(nu=outliers_fraction, shuffle=True, fit_intercept=True, random_state=42, tol=1e-4),
            "Isolation Forest": IsolationForest(contamination=outliers_fraction,random_state=42),
            "Local Outlier Factor": LOF(n_neighbors=50, contamination=outliers_fraction)
        }

    name = clsname
    clf = classifiers[name]


    try:
        clf.fit(X_train)
        y_pred = clf.predict(X_val)
        val_score = metrics.accuracy_score(y_val,y_pred)
    except Exception as e:
        print(e)
        y_pred = []
        val_score = 0
    
    test_scores = []
    for maltype in test_data:
        X_test = np.array(maltype)
        y_test = [-1  for i in range(0,len(X_test))]

        if X_test.ndim == 3:
            nsamples, nx, ny = X_test.shape
            X_test = X_test.reshape((nsamples,nx*ny))

        y_pred = clf.predict(X_test)
        test_scores.append(metrics.accuracy_score(y_test, y_pred))
    
    X_test = np.array(total_test_data)
    y_test = [-1  for i in range(0,len(X_test))]

    if X_test.ndim == 3:
        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))

    y_pred = clf.predict(X_test)
    test_scores.append(metrics.accuracy_score(y_test, y_pred))
    return clf, val_score, test_scores


def run(argv):
    folder = argv[0]
    dirname = os.path.dirname(__file__)
    csv_files = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/features/csv/"))
    features = os.listdir(csv_files)
    malwares=["delay", "confusion", "freeze", "mimic", "noise", "repeat", "spoof"]
    resultsPath = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/results/"))
    os.makedirs(resultsPath, exist_ok=True)
    clss = ["Isolation Forest", "One-Class SVM",  "Robust covariance","SGD One-Class SVM", "LOF"]
    res = []
    for clsname in clss:
        for feature in features:
            print("Running {} with feature: {}".format(clsname, feature))
            tsv_name = csv_files + "\\"+ feature
            encoded_trace_df = pd.read_csv(tsv_name, sep='\t')
            feature = feature.replace(".csv", "")

            ft = [ast.literal_eval(i) for i in encoded_trace_df[feature]]
            encoded_trace_df[feature] = ft

            train_data = encoded_trace_df[encoded_trace_df.maltype=='normal'][feature].tolist()
            total_test_data, test_data = [], []
            for m in malwares:
                test_data.append(encoded_trace_df[encoded_trace_df.maltype==m][feature].tolist())
                total_test_data += encoded_trace_df[encoded_trace_df.maltype==m][feature].tolist()
            clf, val_score, test_scores = train(train_data, test_data, clsname, total_test_data)
            for i,m in enumerate(malwares):
                res.append((feature, clsname, val_score, m, test_scores[i]))
            res.append((feature, clsname, val_score, 'total', test_scores[-1]))

        
                # clfName = '{}_{}.pk'.format(clsname, feature)
                # loc=open(resultsPath + "\\" + clfName,'wb')
                # pickle.dump(clf, loc)
            df = pd.DataFrame(res)
            df.to_csv(resultsPath + "\\" + 'res.csv', index=False)

    df = pd.DataFrame(res)
    df.columns = ['feature', 'clsname', 'val_score', 'malware', 'test_score']
    print(df)
    df.to_csv(resultsPath + "\\" + 'res.csv', index=False)
 

if __name__ == "__main__":
    run(sys.argv[1:])