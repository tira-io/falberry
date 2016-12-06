import argparse
import pickle
from datetime import datetime

from sklearn.preprocessing import LabelEncoder

cat_features = ['userName', 
                 'contentType', 'revisionAction', 'revisionPrevAction', 'revisionSubaction','revisionLanguage',
                 'userCity', 'userContinent', 'userCountry', 'userCounty', 'userRegion', 'userTimeZone', 'revisionTag',
                  'param1', 
                 'param3', 'param4', 'property'
                #'commentTail', 'englishItemLabel','latestEnglishItemLabel',
                 ]

# first step encode data
def build_encode_data(input_file, output_folder):
    print 'Loading data...'
    encode_data = pickle.load(open(input_file, 'rb'))
    for key in cat_features:
        print 'Encoding for ' + key
        le = LabelEncoder()
        data = encode_data[key]
        data.add('missing')
        data = list(data)
        print len(data)

        le.fit(data)
        with open(output_folder + key + ".pickle", 'wb') as f:
            pickle.dump(le, f, -1)
        del encode_data[key]

def main():
    start = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', required=False, dest='input_file',
                        default='../input/train_feature_all_encode_data.pickle')
    parser.add_argument('--output-file', '-o', required=False, dest='output_file',
                        default='../models_encode_v0/')
    args = parser.parse_args()
    build_encode_data(args.input_file, args.output_file)
    print ("Finish - " + str(datetime.now() - start))


if __name__ == "__main__":
    main()
