from pathlib import Path
import pandas as pd

PATH = Path(__file__).parent


# Data.csv to pandas dataframe
def read_csv(file_name: str = "data.csv") -> pd.DataFrame:
    """
    Read a csv file and return a pandas dataframe
    :param file_name: name of the csv file
    :return: pandas dataframe
    """
    df = pd.read_csv(PATH / file_name)
    return df


# From pandas to list of string (concatenate all columns)
def from_pandas_to_list(df: pd.DataFrame, separator: str = '\n') -> list:
    """
    Convert a pandas dataframe to a list of strings
    :param df: pandas dataframe
    :param separator: separator to use to concatenate the columns
    :return: list of strings
    """
    df = df.apply(lambda x: x.astype(str).str.cat(sep=separator), axis=1)
    return df.tolist()

