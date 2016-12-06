import argparse
import logging
import warnings

import pandas as pd
import xgboost as xgb
import feature_extraction_v0
warnings.filterwarnings("ignore")

goal = 'ROLLBACK_REVERTED'
myid = 'revisionId'


def xgb_test(input_file, model_file, encode_file, predict_file):
    logging.info("Loading test data...")
    test = pd.read_csv(input_file, compression='infer', usecols = feature_extraction.use_cols)

    logging.info("Loading model...")
    gbm = xgb.Booster(model_file=model_file)

    logging.info("Predicting...")
    ids = test[myid]
    del test[myid]
    test, features = feature_extraction_v0.process(test, encode_file)
    test_probs = gbm.predict(xgb.DMatrix(test))
    submission = pd.DataFrame({myid: ids, goal: test_probs})
    submission.to_csv(predict_file, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', required=False, dest='in_file', default='../input/wdvc16_2012_10.features.bz2')
    parser.add_argument('--model-file', '-m', required=False, dest='model_file', default='../models/xgb_model_v0.bin')
    parser.add_argument('--encode-file', '-e', required=False, dest='encode_file', default='../models_encode_v0/')
    parser.add_argument('--output-file', '-o', required=False, dest='out_file', default='../submissions/wdvc16_2012_10.submissions_v0.csv')
    args = parser.parse_args()
    xgb_test(args.in_file, args.model_file, args.encode_file, args.out_file)
    #xgb_test(args.in_file, args.model_file, args.encode_file, args.out_file)


if __name__ == "__main__":
    main()
