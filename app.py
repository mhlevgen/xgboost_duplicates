import os
import pickle
import time
import pandas as pd
import xgboost as xgb
from flask import Flask, request

from config import MODELS_DIR, PREPROCESSED_DATA_DIR
from src.inference_preprocess import PreprocessInput

app = Flask(__name__)

model = pickle.load(open(os.path.join(MODELS_DIR,
                                      "model_colsample_bytree_0.98_gamma_0.84_max_depth_13_min_child_weight_6.0_subsample_0.58.bin"), "rb"))
one_hot_encoder = pickle.load(open(os.path.join(PREPROCESSED_DATA_DIR,
                                                "one_hot_encoder.bin"), "rb"))
input_preprocessor = PreprocessInput(one_hot_encoder=one_hot_encoder)


@app.route('/predict', methods=['POST'])
def hello():
    start = time.time()

    input_df = pd.DataFrame(request.json)
    api_array = input_preprocessor.preprocess_input(df=input_df)
    offer_id_array = input_preprocessor.get_offer_ids(df=input_df)

    api_array = xgb.DMatrix(api_array, feature_names=model.feature_names)
    predictions = model.predict(api_array, ntree_limit=model.best_iteration + 1)
    response = {'num_pairs': offer_id_array.shape[0],
                'results': []}
    for offer_ids, pred in zip(offer_id_array, predictions):
        response['results'].append({'offer_id_x': offer_ids[0],
                                    'offer_id_y': offer_ids[1],
                                    'probability_duplicates': '{:.3f}'.format(pred)})
    response['elapsed_time'] = '{:.3f} s'.format(time.time() - start)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
