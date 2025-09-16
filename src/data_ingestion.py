import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

os.makedirs('logs', exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

#TO SHOW THE LOGS IN CONSOLE
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_path = os.path.join('logs', 'data_ingestion.log')


#File for the log
file_handler = logging.FileHandler(log_path)
file_handler.setLevel("DEBUG")
file_handler.setFormatter('')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

#type hinting
def data_load(data_file : str) -> pd.DataFrame:
    """ This Function loads the csv file into a Dataframe df. """

    try:
        df = pd.read_csv(data_file)
        logger.debug('The data has been succesfully loaded from %s', data_file)
        return df
    except Exception as e:
        logger.error('Unexpected Error %s while loading the data.', e)
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ This Function cleans the data for further preprocessing. """

    try:
        df = df.copy()
        df = df.drop_duplicates(keep='first')
        df = df.iloc[:, :-3]
        df = df.rename(columns= {'v1': 'target_label', 'v2': 'text'})
        logger.debug('The raw data has been cleaned for further preprocessing.')
        return df

    except Exception as e:
        logger.error('Unexpected Error occured during cleaning: %s', e)
        raise

def save_data(store_data: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """ This Function Saves the data into train and test csv files. """
    try:
        #Creating ./data/raw 
        raw_path = os.path.join(store_data, 'raw')
        os.makedirs(raw_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)

        logger.debug('Train and Test data has been saved at %s', raw_path)

    except Exception as e:
        logger.error('Unexpected error occured while saving the data into train and test csv files: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_paths = './experiments/spam.csv'
        df = data_load(data_file=data_paths)
        final_df = clean_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state= 39)
        save_data(store_data = './data', train_data= train_data, test_data = test_data)

    except Exception as e:
        logger.error('Unexpected error occured while running the file: %s', e)
        raise


if __name__ == '__main__':
    main()