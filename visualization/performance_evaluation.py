import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def average(lst):
    return sum(lst) / len(lst)

def main():
    folders = ["2022-07-06_13-06-52_fft_20000_30", "2022-07-07_17-27-05_fft_200000_30", "2022-07-08_22-41-52_rtlsdr_20000000_30", "2022-07-10_04-08-51_rtlsdr_200000000_30", "2022-07-11_16-41-53_rtlsdr_800000000_30"]
    bandwidths = ["20000", "200000", "20000000", "200000000", "800000000"]
    models = ["Isolation Forest", "One-Class SVM",  "Robust covariance","SGD One-Class SVM", "Local Outlier Factor"]
    features = ["countvectorizer_ngram1", "countvectorizer_ngram2", "countvectorizer_ngram3", "hashingvectorizer_ngram1", "index_sequence_features", "onehot_sequence_features", "tfidfvectorizer_ngram1", "tfidfvectorizer_ngram2", "tfidfvectorizer_ngram3"]
    modes = ["repeat", "mimic", "confusion", "noise", "spoof", "freeze", "delay", "total"]
    dirname = os.path.dirname(__file__)
    modes_folder = os.path.abspath(os.path.join(dirname, "modes"))
    graphics_folder = os.path.abspath(os.path.join(dirname, "graphics"))
    arr_tpr = np.zeros((len(folders), len(models), len(features), len(modes)))
    arr_tnr = np.zeros((len(folders), len(models), len(features), len(modes)))

    for i, folder in enumerate(folders):
        results_path = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/results/"))
        results = pd.read_csv(os.path.join(results_path, "res.csv"), sep=',')
        for index, row in results.iterrows():
            feature, model, val_score, mode, performance = row[0], row[1], row[2], row[3], row[4]
            if val_score >= 0.87:
                arr_tpr[i, models.index(model), features.index(feature), modes.index(mode)] = performance
            else: 
                arr_tpr[i, models.index(model), features.index(feature), modes.index(mode)] = 0
    print(arr_tpr)

    for i, folder in enumerate(folders):
        results_path = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/results/"))
        results = pd.read_csv(os.path.join(results_path, "res.csv"), sep=',')
        for index, row in results.iterrows():
            feature, model, val_score, mode, performance = row[0], row[1], row[2], row[3], row[4]
            arr_tnr[i, models.index(model), features.index(feature), modes.index(mode)] = val_score

    print(arr_tnr)

    # -------------------------------------------------------------------------------- #
    # all models over all bandwidths - TPR
    models_total_average_tpr = []
    for i, folder in enumerate(folders): #folders
        for model in models: #models
            all_features_average_tpr = []
            for feature in features: #features
                all_features_average_tpr.append(arr_tpr[i, models.index(model), features.index(feature), modes.index("total")])
                print("The " + model + " algorithm has a performance of " + str(arr_tpr[i, models.index(model), features.index(feature), modes.index("total")]) + " with the feature " + feature)
                #for mode in modes: #modes
                    #performance
            models_total_average_tpr.append(average(all_features_average_tpr))
            print("The " + model + " algorithm has an average performance of " + str(average(all_features_average_tpr))+ " with a bandwidth of: " + bandwidths[i])

    IF_performances_tpr = pd.DataFrame(list(zip(5*["Isolation Forest"],bandwidths,models_total_average_tpr[models.index("Isolation Forest")::len(models)])))
    SVM_performances_tpr = pd.DataFrame(list(zip(5*["One-Class SVM"],bandwidths, models_total_average_tpr[models.index("One-Class SVM")::len(models)])))
    RC_performances_tpr = pd.DataFrame(list(zip(5*["Robust covariance"],bandwidths, models_total_average_tpr[models.index("Robust covariance")::len(models)])))
    SGDSVM_performances_tpr = pd.DataFrame(list(zip(5*["SGD One-Class SVM"],bandwidths,models_total_average_tpr[models.index("SGD One-Class SVM")::len(models)])))
    LOF_performances_tpr =  pd.DataFrame(list(zip(5*["Local Outlier Factor"],bandwidths, models_total_average_tpr[models.index("Local Outlier Factor")::len(models)])))

    df_mod_bw_tpr = pd.concat([IF_performances_tpr, SVM_performances_tpr, RC_performances_tpr, SGDSVM_performances_tpr, LOF_performances_tpr])
    df_mod_bw_tpr.columns =['model', 'bandwidth', 'performance']
    print(df_mod_bw_tpr)

    plt.figure()
    ax = plt.gca()  
    df_mod_bw_fig = sns.barplot(data=df_mod_bw_tpr, x='bandwidth', y='performance', hue='model')
    ax.set_ylabel('TPR')
    ax.set_xlabel('')
    plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Model_bandwidth_TPR_comparison.png", bbox_inches="tight")))

    # -------------------------------------------------------------------------------- #
    # average models & bandwidhts - TPR 
    IF_total_tpr_average = average(models_total_average_tpr[models.index("Isolation Forest")::len(models)])
    SVM_total_tpr_average = average(models_total_average_tpr[models.index("One-Class SVM")::len(models)])
    RC_total_tpr_average = average(models_total_average_tpr[models.index("Robust covariance")::len(models)])
    SGDSVM_total_tpr_average = average(models_total_average_tpr[models.index("SGD One-Class SVM")::len(models)])
    LOF_total_tpr_average = average(models_total_average_tpr[models.index("Local Outlier Factor")::len(models)])

    df_mod_tpr = pd.DataFrame(columns=["model", "performance"])
    df_mod_tpr.loc[len(df_mod_tpr.index)] = ["Isolation Forest", IF_total_tpr_average]
    df_mod_tpr.loc[len(df_mod_tpr.index)] = ["One-Class SVM", SVM_total_tpr_average]
    df_mod_tpr.loc[len(df_mod_tpr.index)] = ["Robust covariance", RC_total_tpr_average]
    df_mod_tpr.loc[len(df_mod_tpr.index)] = ["SGD One-Class SVM", SGDSVM_total_tpr_average]
    df_mod_tpr.loc[len(df_mod_tpr.index)] = ["Local Outlier Factor", LOF_total_tpr_average]

    print(df_mod_tpr)

    plt.figure()
    ax = plt.gca()
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    df_mod_fig = sns.barplot(data=df_mod_tpr, x='model', y='performance')
    ax.set_ylabel('TPR')
    ax.set_xlabel('')
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Model_TPR_comparison.png", bbox_inches="tight")))

    # -------------------------------------------------------------------------------- #
    # all models over all bandwidths - TNR
    models_total_average_tnr = []
    for i, folder in enumerate(folders): #folders
        for model in models: #models
            all_features_average_tnr = []
            for feature in features: #features
                all_features_average_tnr.append(arr_tnr[i, models.index(model), features.index(feature), modes.index("total")])
                print("The " + model + " algorithm has a performance of " + str(arr_tnr[i, models.index(model), features.index(feature), modes.index("total")]) + " with the feature " + feature)
                #for mode in modes: #modes
                    #performance
            models_total_average_tnr.append(average(all_features_average_tnr))
            print("The " + model + " algorithm has an average performance of " + str(average(all_features_average_tnr))+ " with a bandwidth of: " + bandwidths[i])

    IF_performances_tnr = pd.DataFrame(list(zip(5*["Isolation Forest"],bandwidths,models_total_average_tnr[models.index("Isolation Forest")::len(models)])))
    SVM_performances_tnr = pd.DataFrame(list(zip(5*["One-Class SVM"],bandwidths, models_total_average_tnr[models.index("One-Class SVM")::len(models)])))
    RC_performances_tnr = pd.DataFrame(list(zip(5*["Robust covariance"],bandwidths, models_total_average_tnr[models.index("Robust covariance")::len(models)])))
    SGDSVM_performances_tnr = pd.DataFrame(list(zip(5*["SGD One-Class SVM"],bandwidths,models_total_average_tnr[models.index("SGD One-Class SVM")::len(models)])))
    LOF_performances_tnr =  pd.DataFrame(list(zip(5*["Local Outlier Factor"],bandwidths, models_total_average_tnr[models.index("Local Outlier Factor")::len(models)])))

    df_mod_bw_tnr = pd.concat([IF_performances_tnr, SVM_performances_tnr, RC_performances_tnr, SGDSVM_performances_tnr, LOF_performances_tnr])
    df_mod_bw_tnr.columns =['model', 'bandwidth', 'performance']
    print(df_mod_bw_tnr)

    plt.figure()
    df_mod_bw_fig = sns.barplot(data=df_mod_bw_tnr, x='bandwidth', y='performance', hue='model')
    ax.set_ylabel('TNR')
    ax.set_xlabel('')    
    plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Model_bandwidth_TNR_comparison.png", bbox_inches="tight")))

    # -------------------------------------------------------------------------------- #
    # average models & bandwidhts - TNR
    IF_total_average_tnr = average(models_total_average_tnr[models.index("Isolation Forest")::len(models)])
    SVM_total_average_tnr = average(models_total_average_tnr[models.index("One-Class SVM")::len(models)])
    RC_total_average_tnr = average(models_total_average_tnr[models.index("Robust covariance")::len(models)])
    SGDSVM_total_average_tnr = average(models_total_average_tnr[models.index("SGD One-Class SVM")::len(models)])
    LOF_total_average_tnr = average(models_total_average_tnr[models.index("Local Outlier Factor")::len(models)])

    df_mod_tnr = pd.DataFrame(columns=["model", "performance"])
    df_mod_tnr.loc[len(df_mod_tnr.index)] = ["Isolation Forest", IF_total_average_tnr]
    df_mod_tnr.loc[len(df_mod_tnr.index)] = ["One-Class SVM", SVM_total_average_tnr]
    df_mod_tnr.loc[len(df_mod_tnr.index)] = ["Robust covariance", RC_total_average_tnr]
    df_mod_tnr.loc[len(df_mod_tnr.index)] = ["SGD One-Class SVM", SGDSVM_total_average_tnr]
    df_mod_tnr.loc[len(df_mod_tnr.index)] = ["Local Outlier Factor", LOF_total_average_tnr]

    print(df_mod_tnr)

    plt.figure()
    ax = plt.gca()
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    df_mod_fig = sns.barplot(data=df_mod_tnr, x='model', y='performance')
    ax.set_ylabel('TNR')
    ax.set_xlabel('')
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Model_TNR_comparison.png", bbox_inches="tight")))

    # -------------------------------------------------------------------------------- #
    # average models & bandwidhts
    df_mod_tnr["Accuracy"] = 5*["TNR"]
    df_mod_tpr["Accuracy"] = 5*["TPR"]

    df_mod_tnr_tpr = pd.concat([df_mod_tnr, df_mod_tpr])

    plt.figure()
    ax = plt.gca()
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    df_mod_tnr_tpr_fig = sns.barplot(data=df_mod_tnr_tpr, x='model', y='performance', hue='Accuracy')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Model_TPR_TNR_comparison.png", bbox_inches="tight")))

    # -------------------------------------------------------------------------------- #
    # all bws and modes for RC with tfidfvectorizer_ngram1
    attack_comparison = []
    for i, folder in enumerate(folders): #folders
        for mode in modes: #features
            attack_comparison.append(arr_tpr[i, models.index("Robust covariance"), features.index("tfidfvectorizer_ngram1"), modes.index(mode)])
            print("The " + model + " algorithm has a performance of " + str(arr_tpr[i, models.index("Robust covariance"), features.index("tfidfvectorizer_ngram1"), modes.index(mode)]) + " with the feature " + feature + " in mode " + mode)

    RC_TFIDF_modes_20000 = pd.DataFrame(list(zip(len(modes)*["20000"],modes,attack_comparison[bandwidths.index("20000")*len(modes):bandwidths.index("20000")+len(modes)])))
    RC_TFIDF_modes_200000 = pd.DataFrame(list(zip(len(modes)*["200000"],modes, attack_comparison[bandwidths.index("200000")*len(modes):bandwidths.index("200000")*len(modes)+len(modes)])))
    RC_TFIDF_modes_20000000 = pd.DataFrame(list(zip(len(modes)*["20000000"],modes, attack_comparison[bandwidths.index("20000000")*len(modes):bandwidths.index("20000000")*len(modes)+len(modes)])))
    RC_TFIDF_modes_200000000 = pd.DataFrame(list(zip(len(modes)*["200000000"],modes,attack_comparison[bandwidths.index("200000000")*len(modes):bandwidths.index("200000000")*len(modes)+len(modes)])))
    RC_TFIDF_modes_800000000 = pd.DataFrame(list(zip(len(modes)*["800000000"],modes, attack_comparison[bandwidths.index("800000000")*len(modes):bandwidths.index("800000000")*len(modes)+len(modes)])))

    RC_TFIDF_modes_bw = pd.concat([RC_TFIDF_modes_20000, RC_TFIDF_modes_200000, RC_TFIDF_modes_20000000, RC_TFIDF_modes_200000000, RC_TFIDF_modes_800000000])
    RC_TFIDF_modes_bw.columns =['bandwidth', 'mode', 'performance']
    print(RC_TFIDF_modes_bw)

    plt.figure()
    RC_TFIDF_modes_bw_fig = sns.barplot(data=RC_TFIDF_modes_bw, x='bandwidth', y='performance', hue='mode')
    plt.title("Robust covariance with tfidfvectorizer_ngram1 for each bandwidth", weight="bold")
    plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Mode_bandwidth_comparison.png", bbox_inches="tight")))

    # -------------------------------------------------------------------------------- #
    # all features over all bandwidths 
    features_total_average = []
    for i, folder in enumerate(folders): #folders
        for feature in features: #features
            all_models_average = []
            for model in models: #models
                all_models_average.append(arr_tpr[i, models.index(model), features.index(feature), modes.index("total")])
                print(feature + " has a performance of " + str(arr_tpr[i, models.index(model), features.index(feature), modes.index("total")]) + " with the model " + model)
            features_total_average.append(average(all_models_average))
            print(feature + " has an average performance of " + str(average(all_models_average))+ " with a bandwidth of: " + bandwidths[i])

    feature_performances_20000 = pd.DataFrame(list(zip(len(features)*["20000"],features,features_total_average[bandwidths.index("20000")*len(features):bandwidths.index("20000")*len(features)+len(features)])))
    feature_performances_200000 = pd.DataFrame(list(zip(len(features)*["200000"],features, features_total_average[bandwidths.index("200000")*len(features):bandwidths.index("200000")*len(features)+len(features)])))
    feature_performances_20000000 = pd.DataFrame(list(zip(len(features)*["20000000"],features, features_total_average[bandwidths.index("20000000")*len(features):bandwidths.index("20000000")*len(features)+len(features)])))
    feature_performances_200000000 = pd.DataFrame(list(zip(len(features)*["200000000"],features,features_total_average[bandwidths.index("200000000")*len(features):bandwidths.index("200000000")*len(features)+len(features)])))
    feature_performances_800000000 = pd.DataFrame(list(zip(len(features)*["800000000"],features, features_total_average[bandwidths.index("800000000")*len(features):bandwidths.index("800000000")*len(features)+len(features)])))

    df_feat_bw = pd.concat([feature_performances_20000, feature_performances_200000, feature_performances_20000000, feature_performances_200000000, feature_performances_800000000])
    df_feat_bw.columns =['bandwidth', 'feature', 'performance']
    print(df_feat_bw)

    plt.figure()
    df_feat_bw_fig = sns.barplot(data=df_feat_bw, x='bandwidth', y='performance', hue='feature')
    plt.title("All features average for each bandwidth", weight="bold")
    plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Features_bandwidth_comparison.png", bbox_inches="tight")))

    # -------------------------------------------------------------------------------- #
    # average features
    df_features_average = pd.DataFrame(columns=["feature", "performance"])
    for feature in features:
        df_features_average.loc[len(df_features_average.index)] = [feature, average(list(df_feat_bw.loc[df_feat_bw["feature"] == feature, "performance"]))]

    print(df_features_average)

    plt.figure()
    ax = plt.gca()
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    df_features_average_fig = sns.barplot(data=df_features_average, x='feature', y='performance')
    ax.set_ylabel('TPR')
    ax.set_xlabel('')
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Feature_comparison.png")), bbox_inches="tight")

    # # -------------------------------------------------------------------------------- #
    # scaled vs not scaled
    folders_scaled = ["2022-07-07_17-27-05_fft_200000_30"]
    features_scaled = ["countvectorizer_ngram1-scaled", "countvectorizer_ngram2-scaled", "countvectorizer_ngram3-scaled", "tfidfvectorizer_ngram1-scaled", "tfidfvectorizer_ngram2-scaled", "tfidfvectorizer_ngram3-scaled"]
    features_not_scaled = ["countvectorizer_ngram1", "countvectorizer_ngram2", "countvectorizer_ngram3", "tfidfvectorizer_ngram1", "tfidfvectorizer_ngram2", "tfidfvectorizer_ngram3"]

    arr_tpr_scaled = np.zeros((len(folders_scaled), len(models), len(features_scaled), len(modes)))
    for i, folder in enumerate(folders_scaled):
        results_path = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/results/"))
        results = pd.read_csv(os.path.join(results_path, "res-scaled.csv"), sep=',')
        for index, row in results.iterrows():
            feature, model, val_score, mode, performance = row[0], row[1], row[2], row[3], row[4]
            if val_score >= 0.87:
                arr_tpr_scaled[i, models.index(model), features_scaled.index(feature), modes.index(mode)] = performance
            else: 
                arr_tpr_scaled[i, models.index(model), features_scaled.index(feature), modes.index(mode)] = 0

    features_total_average_scaled = []
    features_total_average_not_scaled = []
    for folder in folders_scaled: #folders
        for i, feature in enumerate(features_scaled): #features
            all_models_average_scaled = []
            all_models_average_not_scaled = []
            for model in models: #models
                all_models_average_scaled.append(arr_tpr_scaled[0,models.index(model), features_scaled.index(feature), modes.index("total")])
                all_models_average_not_scaled.append(arr_tpr[folders.index(folder), models.index(model), features.index(features_not_scaled[i]), modes.index("total")])
            features_total_average_scaled.append(average(all_models_average_scaled))
            features_total_average_not_scaled.append(average(all_models_average_not_scaled))
            print(feature + " has an average performance of " + str(features_total_average_scaled[-1]))
            print(features_not_scaled[i] + " has an average performance of " + str(features_total_average_not_scaled[-1]))

    print(features_total_average_scaled)
    print(features_total_average_not_scaled)

    features_scaled_df = pd.DataFrame(list(zip(len(features_total_average_scaled)*["scaled"],features_not_scaled,features_total_average_scaled)))
    features_not_scaled_df = pd.DataFrame(list(zip(len(features_total_average_scaled)*["normal"],features_not_scaled,features_total_average_not_scaled)))

    df_scaled_vs_normal = pd.concat([features_scaled_df ,features_not_scaled_df])
    df_scaled_vs_normal.columns =['scaled', 'feature', 'performance']
    print(df_scaled_vs_normal)

    plt.figure()
    ax = plt.gca()
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    df_scaled_vs_normal_fig = sns.barplot(data=df_scaled_vs_normal, x='feature', y='performance', hue='scaled')
    ax.set_ylabel('TPR')
    ax.set_xlabel('')
    plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    plt.savefig(os.path.abspath(os.path.join(graphics_folder, "Features_scaled_comparison.png", bbox_inches="tight")))

    # -------------------------------------------------------------------------------- #
    # modes over bandwidth
    df_modes_model_bw = pd.DataFrame(columns=["bandwidth", "model", "mode", "performance"])
    modes.append("normal")
    for i, folder in enumerate(folders): #folders
        for model in models:
            for mode in modes: #features
                if mode == "normal":
                    performance = (arr_tnr[i, models.index(model), features.index("countvectorizer_ngram3"), 0])
                    df_modes_model_bw.loc[len(df_modes_model_bw.index)] = [bandwidths[i], model, mode, performance]
                else:
                    performance = (arr_tpr[i, models.index(model), features.index("countvectorizer_ngram3"), modes.index(mode)])
                    df_modes_model_bw.loc[len(df_modes_model_bw.index)] = [bandwidths[i], model, mode, performance]



    print(df_modes_model_bw)

    for mode in modes:
        df_mode = df_modes_model_bw.loc[df_modes_model_bw["mode"] == mode, ["model", "performance"]]
        plt.figure()
        ax = plt.gca()
        ax.set_ylim([-.1, 1.1])
        for model in models:
            plt.plot(bandwidths, list(df_mode.loc[df_mode["model"] == model, "performance"]), label=model)
        
        plt.legend(loc="best")
        if mode != "total":
            plt.title(mode, weight="bold")
        else: plt.title("")
        
        if mode == "normal":
            ax.set_ylabel('TNR')
        else:
            ax.set_ylabel('TPR')
        ax.set_xlabel('')
        plt.savefig(modes_folder + "\\" + mode + ".png", bbox_inches="tight")


if __name__ == "__main__":
    main()
