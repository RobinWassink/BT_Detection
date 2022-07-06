import csv
import os
import re
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def countSysCalls(folder):
    sysCalls = {}
    regex = re.compile('.*csv$')
    for root, dirs, files in os.walk(folder):
        for file in files:
            if regex.match(file):
                with open(folder+"/"+file, "r") as systemCallsFile:
                    csvReader = csv.reader(systemCallsFile, delimiter=",")
                    for row in csvReader:
                        if row[2] not in sysCalls:
                            sysCalls[row[2]] = 1
                        else:
                            sysCalls[row[2]] += 1
        return sysCalls


def main(argv):
    modes = ["normal", "repeat", "mimic", "confusion", "noise", "spoof", "freeze", "delay"]
    sysCallsList = []
    timestamp = argv[0]
    dirname = os.path.dirname(__file__)
    for mode in modes:
        sysCalls = countSysCalls(os.path.abspath(os.path.join(dirname, "../data/"+timestamp+"/"+mode)))
        sysCallsList.append(sysCalls)
    df = pd.DataFrame(sysCallsList)
    df.index = modes
    print(df)
    sns.heatmap(df, linewidths=.5, square=True)
    plt.title("System call heatmap for 60 sec monitoring", weight="bold")
    plt.savefig("visualization/heatmap_"+timestamp+".png", bbox_inches="tight")

    

if __name__ == "__main__":
    main(sys.argv[1:])
