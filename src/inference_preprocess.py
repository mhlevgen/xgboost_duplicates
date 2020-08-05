import os
import pickle
import pandas as pd
import numpy as np
import sklearn

from config import PREPROCESSED_DATA_DIR
from src.create_dataset import TextPreprocessor, get_prop_same_words

one_hot_encoder = pickle.load(open(os.path.join(PREPROCESSED_DATA_DIR,
                                                "one_hot_encoder.bin"), "rb"))


class PreprocessInput:
    def __init__(self, one_hot_encoder: sklearn.preprocessing._encoders.OneHotEncoder):
        self.one_hot_encoder = one_hot_encoder

    @staticmethod
    def _preprocess_number_features(df: pd.core.frame.DataFrame) -> np.ndarray:
        df_result = pd.DataFrame()
        for col in ['lat', 'lng']:
            df_result.loc[:, col] = df.apply(lambda x: abs(x['geo_x']['coordinates'].get(col, np.nan) -
                                                           x['geo_y']['coordinates'].get(col, np.nan)),
                                             axis=1)

        for col in ['totalarea', 'roomscount', 'floornumber']:
            df_result.loc[:, col] = abs(df[f'{col}_x'] - df[f'{col}_y'])

        for col in ['price', 'mortgageAllowed']:
            df_result.loc[:, col] = abs(df['bargainterms_x'].get(col, np.nan) -
                                        df['bargainterms_y'].get(col, np.nan))

        df_result.loc[:, 'floorsCount'] = abs(df['building_x'].get('floorsCount', np.nan) -
                                              df['building_y'].get('floorsCount', np.nan))

        return df_result.values

    def _preprocess_cat_features(self, df: pd.core.frame.DataFrame) -> np.ndarray:
        df_result = df[['category_x', 'category_y']]
        get_currency = lambda x: x.get('currency', np.nan)
        df_result.loc[:, 'currency_x'] = df['bargainterms_x'].apply(get_currency)
        df_result.loc[:, 'currency_y'] = df['bargainterms_y'].apply(get_currency)

        get_material_type = lambda x: x.get('materialType', np.nan)
        df_result.loc[:, 'materialType_x'] = df['building_x'].apply(get_material_type)
        df_result.loc[:, 'materialType_y'] = df['building_y'].apply(get_material_type)

        def get_cat_feat_list(suffix): return [f'category_{suffix}',
                                               f'currency_{suffix}',
                                               f'materialType_{suffix}']

        cat_feat_x = df_result[get_cat_feat_list('x')].fillna('no_value')
        cat_feat_y = df_result[get_cat_feat_list('y')].fillna('no_value')

        one_hot_array_x = self.one_hot_encoder.transform(cat_feat_x)
        one_hot_array_y = self.one_hot_encoder.transform(cat_feat_y)

        return abs(one_hot_array_x - one_hot_array_y)

    @staticmethod
    def _preprocess_text_features(df: pd.core.frame.DataFrame) -> np.ndarray:
        text_preprocessor = TextPreprocessor()
        description_x = df['description_x'].apply(lambda x: text_preprocessor.preprocess(x))
        description_y = df['description_y'].apply(lambda y: text_preprocessor.preprocess(y))

        commom_words_part = []

        for descr_x, descr_y in zip(description_x,
                                    description_y):
            commom_words_part.append(get_prop_same_words(descr_x, descr_y))

        return np.array(commom_words_part).reshape(-1, 1)

    def preprocess_input(self, df: pd.core.frame.DataFrame) -> np.ndarray:
        eval_fun_dict_cols = lambda x: eval(x) if isinstance(x, str) else x
        for col in ['geo', 'building', 'bargainterms']:
            for offer in ['x', 'y']:
                df.loc[:, f'{col}_{offer}'] = df[f'{col}_{offer}'].apply(eval_fun_dict_cols)
        numeric_features = self._preprocess_number_features(df)
        one_hot_features = self._preprocess_cat_features(df)
        text_features = self._preprocess_text_features(df)
        return np.hstack((numeric_features,
                          one_hot_features,
                          text_features))

    @staticmethod
    def get_offer_ids(df: pd.core.frame.DataFrame) -> np.ndarray:
        return df[['offer_id_x', 'offer_id_y']].values
