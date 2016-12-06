import argparse
import datetime
import logging
import operator
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import feature_extraction
warnings.filterwarnings("ignore")

from sklearn import cross_validation, metrics
from matplotlib import pylab as plt
from datetime import datetime
seed = 2016
plot = False

goal = 'ROLLBACK_REVERTED'
myid = 'revisionId'

date_format = "%Y-%m-%dT%H:%M:%SZ"


def load_data(file, truth_file, nrows=-1):
    """
        Load data
    """
    print 'Loading data...'
    if (nrows == -1):
        train = pd.read_csv(file, compression='infer', header = None, index_col= None, names = feature_extraction.header_cols)
    else:
        train = pd.read_csv(file, compression='infer', nrows=nrows, header = None, index_col= None, names = feature_extraction.header_cols)
    #print pd.unique(train['param3'])
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #features_cat = train.select_dtypes(exclude=numerics).columns.tolist()
    #print 'Cat feature: '
    #print pd.unique(train['param4'])
    train = train[feature_extraction.use_cols]
    print train.head(5)
    truth = pd.read_csv(truth_file, compression='infer')
    del truth['UNDO_RESTORE_REVERTED']
    truth = truth.rename(columns = {'REVISION_ID':'revisionId'})
    train = train.merge(truth, how = 'left')
    train_y = train[goal].map(lambda x: 1 if x == 'T' else 0)
    del train[goal]
    return (train, train_y)

def xgb_train(train, train_y, features, model_file, val_output_file):

    depth = 5
    eta = 0.01
    ntrees = 601

    params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "min_child_weight": 1,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1}

    print "Running with params: " + str(params)
    print "Running with ntrees: " + str(ntrees)
    print "Running with " + str(len(features)) + " features ..."

    # Training / Cross Validation
    global  seed
    print ("Seed") + str(seed)
    kf = cross_validation.StratifiedKFold(train_y, 4, shuffle=True, random_state=(int)(seed))
    results = []
    predictions = []

    for i, (train_indices, test_indices) in enumerate(kf):
        xgbtrain = xgb.DMatrix(train.iloc[train_indices][features], label=train_y.iloc[train_indices])
        classifier = xgb.train(params, xgbtrain, ntrees)

        y_true = train_y.iloc[test_indices]
        y_predict = classifier.predict(xgb.DMatrix(train.iloc[test_indices][features]))
        indices = y_predict < 0
        y_predict[indices] = 0

        score = metrics.roc_auc_score(y_true, y_predict)
        print ("auc, fold #%d: %f" % (i, score))
        results.append(score)
        # making train prediction (concat with index on the training set - for ensembling purpose)
        pred_df = pd.DataFrame()
        pred_df[myid] = train.iloc[test_indices][myid]
        pred_df[goal] = y_predict
        predictions.insert(0, pred_df)

    print "Results: " + str(results)
    print "Mean: " + str(np.array(results).mean())
    print "Std: " + str(np.array(results).std())

    predictions_df = pd.concat(predictions)
    predictions_df = predictions_df.sort(columns=myid, ascending=True)
    predictions_df.to_csv(val_output_file, index=False)

    # EVAL OR EXPORT
    print str(datetime.now())
    xgbtrain = xgb.DMatrix(train[features], train_y)
    classifier = xgb.train(params, xgbtrain, ntrees)
    classifier.save_model(model_file)

    # Feature importance
    if plot:
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()
        importance = classifier.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        # Plotitup
        plt.figure()
        df.plot()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 150))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.gcf().savefig('Feature_Importance_xgb.png')

def main():
    start = datetime.now()
    print ("Start - " + str(datetime.now()))
    parser = argparse.ArgumentParser()    
    parser.add_argument('--seed', '-s', required=False, dest='seed', default='2016')
    parser.add_argument('--input-file', '-i', required=False, dest='input_file', default='../input/wdvc16_2016_01.features.bz2')
    parser.add_argument('--truth-file', '-t', required=False, dest='truth_file', default='../input/wdvc16_truth.csv.bz2')
    parser.add_argument('--encode-file', '-e', required=False, dest='encode_file', default='../models_encode_v0/')#../input/train_feature_all_encode_data.pickle')
    parser.add_argument('--model-file', '-m', required=False, dest='model_file', default='../models/xgb_model_v0.bin')
    parser.add_argument('--predict-file', '-p', required=False, dest='predict_file',default='../models/xgb_model_v0.val')
    parser.add_argument('--plot', '-pl', required=False, dest='plot', default=False)

    args = parser.parse_args()
    global plot
    plot = args.plot
    global seed
    seed = args.seed
    logging.info("Loading training data...")
    train, train_y = load_data(args.input_file, args.truth_file)
    logging.info("Processing training data...")
    #train, features = feature_extraction.process(train, "non")
    train, features = feature_extraction.process(train, args.encode_file)
    logging.info("Training...")
    xgb_train(train, train_y, features, args.model_file, args.predict_file)
    print ("Finish - " + str(datetime.now() - start))

if __name__ == "__main__":
    main()
