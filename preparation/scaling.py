from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os,sys
import pickle
import ast

# Training data creation for the scaler
def read_feature_tsv(dataPath, ftname):
    tsv_name = os.path.abspath(os.path.join(dataPath, ftname+'.csv'))
    tsv_df = pd.read_csv(tsv_name, sep='\t')
    feature = [ast.literal_eval(i) for i in tsv_df[ftname]]
    tsv_df[ftname] = feature
    normal = tsv_df[tsv_df.maltype=='normal'][ftname].tolist() 
    X_train, X_val = train_test_split(normal, test_size=.3, shuffle=False) 
    return X_train, X_val, tsv_df

def normalize(argv):
    # Define folders needed later on
    folder = argv[0]
    dirname = os.path.dirname(__file__)
    csv_files = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/features/csv-scaled/"))
    features = os.listdir(csv_files)

    # for all the possible bag-of-words features, create a scaler and transform them
    for feature in features:
        if any(tbn in feature for tbn in ["tfidf", "count"]):
            ftname = feature.replace(".csv", "")
            X_train, X_val, tsv_df = read_feature_tsv(csv_files, ftname)
            scaler = StandardScaler().fit(X_train)
            data = scaler.transform(tsv_df[ftname].tolist())
            normaled_df = pd.DataFrame([tsv_df['ids'].tolist(), tsv_df['maltype'].tolist(), data.tolist()]).transpose()
            normaled_df.columns = ['ids', 'maltype', ftname + '-scaled']
            normaled_df.to_csv(os.path.abspath(os.path.join(csv_files, "{}-scaled.csv".format(ftname) + ".pk")), sep='\t', index=None)


if __name__ == "__main__":
    normalize(sys.argv[1:])