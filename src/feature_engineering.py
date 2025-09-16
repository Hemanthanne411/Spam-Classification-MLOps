import logging
import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs('logs', exist_ok=True)

logger = logging.getLogger("preprocessing")
logger.setLevel("DEBUG")

# TO SHOW THE LOGS IN CONSOLE
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_path = os.path.join('logs', 'preprocessing.log')

# File for the log
file_handler = logging.FileHandler(log_path)
file_handler.setLevel("DEBUG")

# Specifying the logging message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# avoid duplicate handlers (useful if script re-runs in dev)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------- Data loading ----------
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file and fill NaN with empty string."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # <-- ensures no np.nan in text
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


# ---------- TF-IDF ----------
def tfidf_vectorize(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Ensure text is string (safety against numbers/None)
        X_train = train_data['text'].astype(str).values
        y_train = train_data['target_label'].values
        X_test = test_data['text'].astype(str).values
        y_test = test_data['target_label'].values

        # fit only on train, transform both
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # give feature names to columns
        feature_names = vectorizer.get_feature_names_out()
        train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
        train_df['target_label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)
        test_df['target_label'] = y_test

        logger.debug('TF-IDF applied and data transformed')
        return train_df, test_df

    except Exception as e:
        logger.error('Unexpected error occurred while vectorizing the data: %s', e)
        raise


# ---------- Save ----------
def save_data(store_data: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save the preprocessed data in a processed folder."""
    try:
        processed_path = os.path.join(store_data, 'processed')
        os.makedirs(processed_path, exist_ok=True)

        train_data.to_csv(os.path.join(processed_path, 'train_tfidf.csv'), index=False)
        test_data.to_csv(os.path.join(processed_path, 'test_tfidf.csv'), index=False)
        logger.debug(f'Preprocessed TF-IDF vectorized data saved at {processed_path}')

    except Exception as e:
        logger.error('Unexpected error occurred while saving: %s', e)
        raise


# ---------- Main ----------
def main():
    try:
        max_features = 50

        # use load_data (ensures NaN handling)
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = tfidf_vectorize(train_data, test_data, max_features)

        save_data("./data", train_df, test_df)

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()