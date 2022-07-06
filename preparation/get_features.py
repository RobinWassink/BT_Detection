import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import os,sys
import tqdm
import pickle
import time
import numpy as np
import csv

def get_syscall_dict(ngrams_dict):
    syscall_dict = {}
    i = 0
    for ngram in ngrams_dict:
        if len(ngram.split()) == 1:
            syscall_dict[ngram] = i
            i+=1
    return syscall_dict

def create_vectorizers(corpus, base_dict_path):
    os.mkdir(base_dict_path)
    for i in range(1, 4):
        cvName = 'countvectorizer_ngram{}.pk'.format(i)
        tvName = 'tfidfvectorizer_ngram{}.pk'.format(i)
        hvName = 'hashingvectorizer_ngram{}.pk'.format(i)
        ndName = 'ngrams_dict_ngram{}.pk'.format(i)
        sdName = 'syscall_dict_ngram{}.pk'.format(i)


        countvectorizer = CountVectorizer(ngram_range=(1, i)).fit(corpus)
        pickle.dump(countvectorizer, open(base_dict_path + "\\" + cvName, "wb"))

        ngrams_dict = countvectorizer.vocabulary_
        pickle.dump(ngrams_dict, open(base_dict_path + "\\" + ndName, "wb"))

        tfidfvectorizer = TfidfVectorizer(ngram_range=(1, i), vocabulary=ngrams_dict).fit(corpus)
        pickle.dump(tfidfvectorizer, open(base_dict_path + "\\" + tvName, "wb"))

        if i == 1:
            syscall_dict = get_syscall_dict(ngrams_dict)
            pickle.dump(syscall_dict, open(base_dict_path + "\\" + sdName, "wb"))

            hashingvectorizer = HashingVectorizer(n_features=2**5).fit(corpus)  
            pickle.dump(hashingvectorizer, open(base_dict_path + "\\" + hvName, "wb"))
        

def from_trace_to_longstr(syscall_trace):
    tracestr = ''
    for syscall in syscall_trace:
        tracestr += syscall + ' '
    # print(tracestr)
    return tracestr


def read_all_rawdata(rawdataPath, rawFileNames):
    corpus_dataframe, corpus = [],[]
    if any('.pk' in fileName for fileName in rawFileNames):
        print("reading data from pickle files")
        loc=open(rawdataPath + "\\" + 'corpus_dataframe.pk','rb')
        corpus_dataframe = pickle.load(loc)
        loc=open(rawdataPath + "\\" + 'corpus.pk','rb')
        corpus = pickle.load(loc)
    else:
        print("reading raw data")
        par = tqdm.tqdm(total=len(rawFileNames), ncols=100)
        for fn in rawFileNames:
            if '.csv' in fn:
                par.update(1)
                fp = rawdataPath + "/" + fn
                trace = pd.read_csv(fp)
                tr = trace['syscall'].tolist()             
                longstr = from_trace_to_longstr(tr)
                corpus_dataframe.append(trace)
                corpus.append(longstr)
        pickle.dump(corpus, open(rawdataPath + "\\" + 'corpus.pk', "wb"))
        pickle.dump(corpus_dataframe, open(rawdataPath + "\\" + 'corpus_dataframe.pk', "wb"))
        par.close()
    return corpus_dataframe, corpus


def create_onehot_encoding(total, index):
    onehot = []
    for i in range(0, total):
        if i == index:
            onehot.append(1)
        else:
            onehot.append(0)
    return onehot

def add_unk_to_dict(syscall_dict):
    total = len(syscall_dict)
    syscall_dict['unk'] = total
    syscall_dict_onehot = dict()
    for sc in syscall_dict:
        syscall_dict_onehot[sc] = create_onehot_encoding(total+1, syscall_dict[sc])
    return syscall_dict, syscall_dict_onehot


def replace_with_unk(syscall_trace, syscall_dict):
    for i, sc in enumerate(syscall_trace):
        if sc.lower() not in syscall_dict:
            syscall_trace[i] = 'unk'
    return syscall_trace

def trace_onehot_encoding(trace, syscall_dict_onehot):
    encoded_trace = []
    for syscall in trace:
        syscall = syscall.lower()
        if syscall.lower() in syscall_dict_onehot:
            one_hot = syscall_dict_onehot[syscall]
        else:
            syscall = 'UNK'
            one_hot = syscall_dict_onehot[syscall]
        encoded_trace.append(one_hot)
    return encoded_trace

def find_all_head(trace, head):
    starts, ends,se = [], [], []

    for i,s in enumerate(trace):
        if s == head:
            start=i
            starts.append(start)
            if len(starts) > 1:
                end = starts[-1] 
                ends.append(end)
        if i == len(trace)-1:
            end = len(trace)
            ends.append(end)
    se = [(starts[i], ends[i]) for i in range(0, len(starts))]
    return se


def get_dict_sequence(trace,term_dict):
    dict_sequence = []
    for syscall in trace:
        if syscall in term_dict:
            dict_sequence.append(term_dict[syscall])
        else:
            dict_sequence.append(term_dict['unk'])
    return dict_sequence


def write_to_csv(encoded_trace_df, feature_path):
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    for i in range(2, len(encoded_trace_df.columns)):
        file_name = encoded_trace_df.columns[i] + ".csv"
        df = encoded_trace_df[["ids", "maltype", encoded_trace_df.columns[i]]]
        df.to_csv(feature_path + "\\" + file_name, sep=",", index=False)


def read_dicts(dictPath):
    vectorizers = {}
    dicts = {}
    for i in range(1, 4):
        cvName = 'countvectorizer_ngram{}'.format(i)
        tvName = 'tfidfvectorizer_ngram{}'.format(i)
        hvName = 'hashingvectorizer_ngram{}'.format(i)
        ndName = 'ngrams_dict_ngram{}'.format(i)
        sdName = 'syscall_dict_ngram{}'.format(i)

        loc=open(dictPath + "\\" + cvName+'.pk','rb')
        cv = pickle.load(loc)
        vectorizers[cvName] = cv

        loc=open(dictPath + "\\" + tvName+'.pk','rb')
        tv = pickle.load(loc)
        vectorizers[tvName] = tv

        loc=open(dictPath + "\\" + ndName+'.pk','rb')
        nd = pickle.load(loc)
        dicts[ndName] = nd

        if i == 1:
            loc=open(dictPath + "\\" + hvName+'.pk','rb')
            hv = pickle.load(loc)
            vectorizers[hvName] = hv

            loc=open(dictPath + "\\" + sdName+'.pk','rb')
            sd = pickle.load(loc)
            dicts[sdName] = sd

    return vectorizers, dicts


def get_features(argv):
    folder = argv[0]
    dirname = os.path.dirname(__file__)
    rawdataPath = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/raw/"))
    rawFileNames = os.listdir(rawdataPath)
    times = {} 
    features, columns = [], []
    ids, maltype = [], []
    for fi in rawFileNames:
        if '.csv' in fi:
            fis = fi.split('_')
            fn = fis[0]
            i = '{}_{}_{}'.format(fis[0], fis[1], fis[2])
            maltype.append(fn)
            ids.append(i)

    features.append(ids)
    columns.append("ids")
    features.append(maltype)
    columns.append("maltype")

    print('start to read rawdata')
    corpus_dataframe, corpus = read_all_rawdata( rawdataPath, rawFileNames)
    print('got rawdata')

    base_dict_path = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/features/dict/"))
    if not os.path.exists(base_dict_path):
        print("creating vectorizers")
        create_vectorizers(corpus, base_dict_path)
    print("loading vectorizers")
    vectorizers, dicts = read_dicts(base_dict_path)
    print('get dicts finished!')

    # for key, value in vectorizers.items():
    #     print(key, ' : ', value)
    # for key, value in dicts.items():
    #     print(key, ' : ', value)

    for i in range(1, 4):
        cvName = 'countvectorizer_ngram{}'.format(i)
        tvName = 'tfidfvectorizer_ngram{}'.format(i)

        cv = vectorizers[cvName]
        tv = vectorizers[tvName]

        t1 = time.time()
        frequency_features = cv.transform(corpus)
        t2 = time.time()
        key = cvName
        t = t2 - t1
        times[key] = t
        print(key+": "+str(t))
        frequency_features = frequency_features.toarray()

        t1 = time.time()
        tfidf_features = tv.transform(corpus)
        t2 = time.time()
        t = t2 - t1
        key = tvName
        times[key] = t
        print(key+": "+str(t))
        tfidf_features = tfidf_features.toarray()

        features.append(frequency_features)
        columns.append(cvName)
        features.append(tfidf_features)
        columns.append(tvName)


    hvName = 'hashingvectorizer_ngram{}'.format(1)
    hv = vectorizers[hvName]
    t1 = time.time()
    hashing_features = hv.transform(corpus)
    t2 = time.time()
    t = t2 - t1
    key = hvName
    times[key] = t
    print(key+": "+str(t))
    hashing_features = hashing_features.toarray()
    features.append(hashing_features)     
    columns.append(hvName)

           
    encoded_trace_df = pd.DataFrame(features).transpose()
    encoded_trace_df.columns = columns
    print(encoded_trace_df)
    # resultsPath = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/features/"))
    # encoded_trace_df.to_pickle(resultsPath+'encoded_bow.pkl')  
    feature_path = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/features/csv/"))

    write_to_csv(encoded_trace_df, feature_path)

    return times 


if __name__ == "__main__":
    get_features(sys.argv[1:])