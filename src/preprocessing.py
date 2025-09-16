import logging
import os
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder


os.makedirs('logs', exist_ok=True)

logger = logging.getLogger("preprocessing")
logger.setLevel("DEBUG")

#TO SHOW THE LOGS IN CONSOLE
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_path = os.path.join('logs', 'preprocessing.log')


#File for the log
file_handler = logging.FileHandler(log_path)
file_handler.setLevel("DEBUG")
file_handler.setFormatter('')

#Specifying the logging message forma
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform(text):
    try:
        ps = PorterStemmer()
        stop = stopwords.words('english')
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = [ps.stem(word) for word in text if word not in stop and word not in string.punctuation and word.isalnum()]
        return " ".join(text)

    except Exception as e:
        logger.error('Unexpected Error occured while transforming the text feature: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """ This FUnction applies Label Encoder for target label and then applies transform. """
    try:
        le = LabelEncoder()
        df = df.copy()
        df['target_label'] = le.fit_transform(df['target_label'])

        df['text'] = df['text'].apply(transform)
        logger.debug('PreProcessing with stemming, stopword removal is succesfully completed')
        return df
    except Exception as e:
        logger.error('Unexpected error occured while preprocessing: %s', e)
        raise




def save_data(store_data: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """ This Function saves the preprocessed data in a interim folder """
    try:

        interim_path = os.path.join(store_data, 'interim')
        os.makedirs(interim_path, exist_ok=True)

        train_data.to_csv(os.path.join(store_data, 'train_processed.csv'), index=False)
        test_data.to_csv(os.path.join(store_data, 'test_processed.csv'), index=False)
        logger.debug(f'Preprocessed data saved at {store_data}')

    except Exception as e:
        logger.error('Unexpected error occured while saving: %s', e)
        raise


def main():

    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    store_data = './data/interim'
    preprocessed_train = preprocess_data(train_data)
    preprocessed_test = preprocess_data(test_data)
    save_data(store_data=store_data, train_data=preprocessed_train, test_data=preprocessed_test)


if __name__ == '__main__':
    main()
