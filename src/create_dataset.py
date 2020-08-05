import os
from typing import List, Tuple
import pickle
from string import punctuation

import pandas as pd
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem

from config import PREPROCESSED_DATA_DIR, MODELS_DIR

PATH_TO_NLTK_DATA = os.path.join(MODELS_DIR, 'nltk_data')
nltk.data.path.append(PATH_TO_NLTK_DATA)


class LoadDataFromDb:
    def __init__(self):
        my_connection_string = 'postgresql://postgres:test-task@localhost:8213/postgres'
        self.engine = create_engine(my_connection_string)

    def select(self, query):
        return pd.read_sql(query, self.engine)


class TextPreprocessor:

    def __init__(self):
        self.mystem = Mystem()
        self.russian_stopwords = stopwords.words("russian")

    def preprocess(self, text: str) -> List:
        tokens = self.mystem.lemmatize(text.lower())
        tokens = [token for token in tokens if token not in self.russian_stopwords
                  and token.strip() != ""
                  and all(i not in punctuation for i in token.strip())]

        return tokens


class FeaturePreprocessor:

    def __init__(self,
                 df: pd.core.frame.DataFrame,
                 one_hot_encoder: sklearn.preprocessing._encoders.OneHotEncoder = None):
        for col in ['geo', 'building', 'bargainterms']:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.df = df
        self.one_hot_encoder = one_hot_encoder

    @staticmethod
    def _get_cat_features(row: pd.core.series.Series) -> dict:
        return {'category': row['category'],
                'currency': row['bargainterms'].get('currency', np.nan),
                'materialType': row['building'].get('materialType', np.nan)}

    @staticmethod
    def _get_float_features(row: pd.core.series.Series) -> dict:

        return {'lat': row['geo']['coordinates'].get('lat', np.nan),
                'lng': row['geo']['coordinates'].get('lng', np.nan),
                'totalarea': row['totalarea'],
                'roomscount': row['roomscount'],
                'floornumber': row['floornumber'],
                'price': row['bargainterms'].get('price', np.nan),
                'floorsCount': row['building'].get('floorsCount', np.nan),
                'mortgageAllowed': row['bargainterms'].get('mortgageAllowed', np.nan)}

    def get_cat_df(self) -> pd.core.frame.DataFrame:
        cat_feat_df = self.df.apply(lambda x: self._get_cat_features(x),
                                    axis=1,
                                    result_type='expand')
        return cat_feat_df.fillna('no_value')

    def get_num_df(self) -> pd.core.frame.DataFrame:

        num_feat_df = self.df.apply(lambda x: self._get_float_features(x),
                                    axis=1,
                                    result_type='expand')
        return num_feat_df

    def one_hot_encode(self, cat_feat_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        if self.one_hot_encoder is None:
            self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore',
                                                 sparse=False)
            one_hot_feat_array = self.one_hot_encoder.fit_transform(cat_feat_df)

        else:
            one_hot_feat_array = self.one_hot_encoder.transform(cat_feat_df)

        cat_col_names = []
        for cat_name, feat_value_list in zip(cat_feat_df.columns,
                                             self.one_hot_encoder.categories_):
            cat_col_names.extend(['{}_{}'.format(cat_name, feat)
                                  for feat in feat_value_list])

        return pd.DataFrame(one_hot_feat_array, columns=cat_col_names)


class CreatePairedDataset:
    def __init__(self, offer_features_df: pd.core.frame.DataFrame) -> None:
        self.offer_features_df = offer_features_df
        self.diff_columns = [col for col in offer_features_df.columns
                             if col != 'description']

    def _get_diff_for_offers(self,
                             df_offer_1: pd.core.frame.DataFrame,
                             df_offer_2: pd.core.frame.DataFrame) -> np.ndarray:
        diff = (df_offer_1[self.diff_columns] -
                df_offer_2[self.diff_columns])
        return abs(diff)

    def _get_feature_description(self,
                                 df_offer_1: pd.core.frame.DataFrame,
                                 df_offer_2: pd.core.frame.DataFrame) -> np.ndarray:
        commom_words_part = []

        for descr_x, descr_y in zip(df_offer_1['description'],
                                    df_offer_2['description']):
            commom_words_part.append(get_prop_same_words(descr_x, descr_y))

        return np.array(commom_words_part).reshape(-1, 1)

    @staticmethod
    def _get_target(df_offer: pd.core.frame.DataFrame) -> np.ndarray:
        return df_offer['resolution'].astype(int).values

    def calculate_paired_data(self,
                              pairs: pd.core.frame.DataFrame) -> Tuple[np.ndarray,
                                                                       np.ndarray]:

        df_offer_1 = pairs.join(self.offer_features_df, on=['offer_id1'])
        df_offer_2 = pairs.join(self.offer_features_df, on=['offer_id2'])

        diff_features = self._get_diff_for_offers(df_offer_1, df_offer_2)
        description_intersect = self._get_feature_description(df_offer_1, df_offer_2)
        target = self._get_target(df_offer_1)
        return np.hstack((diff_features, description_intersect)), target


def get_prop_same_words(descr_1: List[str], descr_2: List[str]) -> float:
    descr_1, descr_2 = set(descr_1), set(descr_2)
    number_common_words = len(descr_1.intersection(descr_2))
    min_number_words = min(len(descr_1), len(descr_2))

    return number_common_words / min_number_words


def save_file(path, *args):
    np.savez(path, *args)


if __name__ == "__main__":
    tqdm.pandas()

    dataloader = LoadDataFromDb()
    offers_df = dataloader.select('SELECT * FROM offers')
    pairs_df = dataloader.select('SELECT offer_id1, offer_id2, resolution FROM pairs')

    text_preprocessor = TextPreprocessor()
    offers_df['description'] = offers_df['description'].progress_apply(lambda x: text_preprocessor.preprocess(x))

    feature_preprocessor = FeaturePreprocessor(offers_df)
    cat_feat_df = feature_preprocessor.get_cat_df()
    float_feat_df = feature_preprocessor.get_num_df()
    ohe_hot_df = feature_preprocessor.one_hot_encode(cat_feat_df)

    features_df = pd.concat((float_feat_df,
                             ohe_hot_df,
                             offers_df[['description']]), axis=1)
    features_df.set_index(offers_df['offer_id'], inplace=True)

    train_pairs, test_val_pairs = train_test_split(pairs_df, test_size=0.3,
                                                   random_state=1,
                                                   stratify=pairs_df['resolution'])

    val_pairs, test_pairs = train_test_split(test_val_pairs, test_size=0.33,
                                             random_state=1,
                                             stratify=test_val_pairs['resolution'])

    paired_data_creator = CreatePairedDataset(features_df)
    x_train, y_train = paired_data_creator.calculate_paired_data(train_pairs)
    x_val, y_val = paired_data_creator.calculate_paired_data(val_pairs)
    x_test, y_test = paired_data_creator.calculate_paired_data(test_pairs)

    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

    np.savez(os.path.join(PREPROCESSED_DATA_DIR, 'train.npz'), x_train,
             y_train,
             np.array(features_df.columns))
    np.savez(os.path.join(PREPROCESSED_DATA_DIR, 'val.npz'), x_val, y_val)
    np.savez(os.path.join(PREPROCESSED_DATA_DIR, 'test.npz'), x_test, y_test)

    pickle.dump(feature_preprocessor.one_hot_encoder,
                open(os.path.join(PREPROCESSED_DATA_DIR, "one_hot_encoder.bin"), "wb"))
