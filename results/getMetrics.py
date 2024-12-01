import pandas as pd
import sys
import os

def dataframe_to_markdown(df):
    """
    Convert a pandas DataFrame into a Markdown table string.

    Parameters:
        df (pandas.DataFrame): The DataFrame to convert.

    Returns:
        str: The Markdown table as a string.
    """
    # Start with the header row
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    
    # Add the data rows
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(map(str, row)) + " |")
    
    # Combine all parts into the final Markdown string
    markdown_table = "\n".join([header, separator] + rows)
    return markdown_table

if __name__ == "__main__":
    method = sys.argv[1]
    resultFile = os.path.join('./results', method, 'result.csv')
    df = pd.read_csv(resultFile, index_col=0)
    print(dataframe_to_markdown(df))
