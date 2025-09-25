import pandas as pd

def check_duplicate_entries(df, column_name):
    """Check if there are duplicate entries in the dataframe"""
    df = df.dropna(subset=[column_name])
    duplicate_entries = df[df.duplicated(subset=[column_name], keep=False)]
    entry_counts = df[column_name].value_counts()

    if len(duplicate_entries) == 0:
        print("No duplicate entries found")
    else:
        print("Repeated entry values:")
        print(duplicate_entries[['entry']])
        print("\n entry counts:")
        print(entry_counts[entry_counts > 1])
        print(f"There are {len(duplicate_entries)} duplicate entries")


def check_overlapping_entries(df1, df2, column_name1, column_name2):
    """Check if there are overlapping entries in the two dataframes"""
    overlapping_entries = df1[df1[column_name1].isin(df2[column_name2])]
    if len(overlapping_entries) == 0:
        print("No overlapping entries found")
    else:
        print("Overlapping entries:")
        print(overlapping_entries[['entry']])
        print(f"There are {len(overlapping_entries)} overlapping entries")
