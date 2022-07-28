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

# Evaluate the performance of the specific algorithm & feature
def evaluate_performance(total_train_data, test_data, mlAlgorithm, total_test_data):
    train_data = np.array(total_train_data)
    train_outcome = [1 for i in range(0,len(train_data))] 
    outliers_fraction = 0.05

    # transform the data to 2 dimensions if necessary (for sequence features)
    if train_data.ndim == 3:
        nsamples, nx, ny = train_data.shape
        train_data = train_data.reshape((nsamples,nx*ny))

    # Split train data into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_outcome, test_size=.3, shuffle=False)
    y_val = [1 for i in range(0,len(X_val))]

    # The ML algorithms
    classifiers = {
            "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
            "One-Class SVM": OneClassSVM(cache_size=200, gamma='scale', kernel='rbf',nu=0.05,  shrinking=True, tol=0.001,verbose=False),
            "SGD One-Class SVM": SGDOneClassSVM(nu=outliers_fraction, shuffle=True, fit_intercept=True, random_state=42, tol=1e-4),
            "Isolation Forest": IsolationForest(contamination=outliers_fraction,random_state=42),
            "Local Outlier Factor": LOF(n_neighbors=20, contamination=outliers_fraction)
        }
    name = mlAlgorithm
    clf = classifiers[name]

    # Train the model and get TNR by predicting the validation set
    try:
        clf.fit(X_train)
        y_pred = clf.predict(X_val)
        TNR = metrics.accuracy_score(y_val,y_pred)
    except Exception as e:
        print(e)
        y_pred = []
        TNR = 0

    # Go through every malicious behavior seperately and get the TPR by predicting all 300 samples
    TPRs = []
    for maltype in test_data:
        X_test = np.array(maltype)
        y_test = [-1  for i in range(0,len(X_test))]

        # Reshape if needed
        if X_test.ndim == 3:
            nsamples, nx, ny = X_test.shape
            X_test = X_test.reshape((nsamples,nx*ny))

        # Get TPR and add to results
        TPR = clf.predict(X_test)
        TPRs.append(metrics.accuracy_score(y_test, TPR))
    
    # Go through the total_test_data list, which contains all attacks together and predict them to get the "total" value (could have been done by averaging the others as well)
    X_test = np.array(total_test_data)
    y_test = [-1  for i in range(0,len(X_test))]
    if X_test.ndim == 3:
        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))
    TPR = clf.predict(X_test)

    # return the performances (TNR & TPRs)
    TPRs.append(metrics.accuracy_score(y_test, TPR))
    return TNR, TPRs


def run(argv):
    # Create folders and variables used later on 
    folder = argv[0]
    dirname = os.path.dirname(__file__)
    csv_files = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/features/csv/"))
    features = os.listdir(csv_files)
    malwares=["delay", "confusion", "freeze", "mimic", "noise", "repeat", "spoof"]
    resultsPath = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/results/"))
    os.makedirs(resultsPath, exist_ok=True)
    mlAlgorithms = ["Isolation Forest", "One-Class SVM",  "Robust covariance","SGD One-Class SVM", "Local Outlier Factor"]
    res = []

    # Go through every feature file for the specific folder
    for feature in features:
        #Read the feature data and dataframe preparation
        tsv_name = os.path.abspath(os.path.join(csv_files, feature))
        encoded_trace_df = pd.read_csv(tsv_name, sep='\t')
        feature = feature.replace(".csv", "")
        ft = [ast.literal_eval(i) for i in encoded_trace_df[feature]]
        encoded_trace_df[feature] = ft

        # create a train dataset with all the normal samples
        total_test_data, test_data = [], []
        train_data = encoded_trace_df[encoded_trace_df.maltype=='normal'][feature].tolist()

        # create a test dataset with all the malicious samples
        # test_data --> a list of every malicious behavior seperate
        # total_test_data --> all malicious behaviors in one list
        for m in malwares:
            test_data.append(encoded_trace_df[encoded_trace_df.maltype==m][feature].tolist())
            total_test_data += encoded_trace_df[encoded_trace_df.maltype==m][feature].tolist()

        # Evaluate every defined ML algorithm with the corresponding train and test datasets
        for mlAlgorithm in mlAlgorithms:
            # Skip sequence features for Robust covariance due to very long training time (8+ hours)
            if ("sequence" in feature and mlAlgorithm == "Robust covariance"):
                print("Skipping "+ mlAlgorithm + " for " + feature)
                continue

            print("Running {} with feature: {}".format(mlAlgorithm, feature))

            # evaluate performance and add them to the results
            TNR, TPR = evaluate_performance(train_data, test_data, mlAlgorithm, total_test_data)
            for i,m in enumerate(malwares):
                res.append((feature, mlAlgorithm, TNR, m, TPR[i]))
            res.append((feature, mlAlgorithm, TNR, 'total', TPR[-1]))

    # print and save results
    df = pd.DataFrame(res)
    df.columns = ['feature', 'mlAlgorithm', 'TNR', 'malware', 'TPR']
    print(df)
    df.to_csv(os.path.abspath(os.path.join(resultsPath, 'res.csv')), index=False)
    
 

if __name__ == "__main__":
    run(sys.argv[1:])