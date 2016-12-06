from csv import DictReader
import pickle
import argparse
from datetime import datetime
import os

cat_features = ['userName', 'revisionLanguage', 'contentType', 
 'longestCharacterSequence', 'longestWord', 'property', 
 'userCity', 'userContinent', 'userCountry', 'userCounty', 'userRegion', 'userTimeZone', 
 'revisionAction', 'revisionPrevAction', 'revisionSubaction', 'revisionTag',
 'param1', 'param3', 'param4',
 'commentTail', 'englishItemLabel','latestEnglishItemLabel',]
encode_data = {}


# first step encode data
def build_encode_data(input_file, output_file):
	for key in cat_features:
		encode_data.setdefault(key, set())
	for t, row in enumerate(DictReader(open(input_file))):		
		for key in row:
			if key in cat_features:
				value = row[key]
				cat_dictionary = encode_data.get(key)
				cat_dictionary.add(value)
				encode_data[key] = cat_dictionary
                                '''
				if not cat_dictionary.has_key(value):
				#if value not in cat_dictionary:
					max_label = 0
					if len(cat_dictionary) > 0:
						max_label = max(cat_dictionary.values()) + 1
					cat_dictionary[value] = max_label
					encode_data[key] = cat_dictionary'''

		if (t + 1) % 1000000 == 0:
			print "proccessed %d lines..." % (t + 1)
			#break
	with open(output_file, 'wb') as f:
		pickle.dump(encode_data, f, -1)


def main():
	start = datetime.now()
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-file', '-i', required=False, dest='input_file',
						default='../input/features/train_feature_all.csv')
	parser.add_argument('--output-file', '-o', required=False, dest='output_file',
						default='../input/train_feature_all_encode_data.pickle')
	args = parser.parse_args()
	build_encode_data(args.input_file, args.output_file)
	print ("Finish - " + str(datetime.now() - start))


if __name__ == "__main__":
	main()
