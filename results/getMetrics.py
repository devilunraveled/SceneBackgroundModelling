import pandas as pd
import sys
import os

if __name__ == "__main__":
    method = sys.argv[1]
    resultFile = os.path.join('./results', method, 'cm.txt')

    # Open the tsv file from pandas.
    df = pd.read_csv(resultFile, sep='\t', index_col=0)
    print(df)
