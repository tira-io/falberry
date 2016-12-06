import argparse
import datetime
import logging
import ujson
import pickle
import numpy as np
from pandas.io.json import json_normalize
import os
date_format = "%Y-%m-%dT%H:%M:%SZ"
cat_features = [
                 'userName', 'revisionLanguage', 'revisionAction', 'revisionPrevAction', 'revisionSubaction', 'property',
                 #'userCity', 'userContinent', 'userCountry', 'userCounty', 'userRegion', 'userTimeZone', 'revisionTag',
                 #'param3', 'param4', 'contentType', 
                 #'commentTail', 'englishItemLabel','latestEnglishItemLabel'
                ]
'''
use_cols = ['revisionId', 'userId', 'userName', 'groupId', 'timestamp', 'revisionLanguage', 'contentType',
            #'commentTail', 'latestEnglishItemLabel','englishItemLabel',
            'itemId', 'superItemId', 'latestInstanceOfItemId', 'property',
            'alphanumericRatio', 'asciiRatio',
            'bracketRatio', 'digitRatio', 'latinRatio', 'longestCharacterSequence', 'lowerCaseRatio',
            'nonLatinRatio', 'punctuationRatio', 'upperCaseRatio', 'whitespaceRatio', 'arabicRatio',
            'bengaliRatio', 'cyrillicRatio', 'hanRatio', 'brahmiRatio', 'malayalamRatio', 'tamilRatio',
            'teluguRatio', 'badWordRatio', 'containsLanguageWord', 'containsURL', 'languageWordRatio',
            'longestWord', 'lowerCaseWordRatio', 'proportionOfLinksAdded', 'proportionOfQidAdded',
            'upperCaseWordRatio', 'proportionOfLanguageAdded', 'commentCommentSimilarity',
            'commentLabelSimilarity', 'commentTailLength', 'commentSitelinkSimilarity',
            'itemValue', 'literalValue', 'isPrivilegedUser', 'isRegisteredUser',
            #'userCity', 'userContinent', 'userCountry', 'userCounty', 'userRegion', 'userTimeZone','revisionTag',
            'isBotUser', 'hasListLabel', 'isHuman', 'labelCapitalizedWordRatio', 'labelContainsFemaleFirstName',
            'labelContainsMaleFirstName', 'numberOfLabels', 'numberOfDescriptions', 'numberOfAliases', 'numberOfStatements',
            'numberOfSitelinks', 'numberOfQualifiers', 'numberOfReferences', 'numberOfBadges', 'isLivingPerson',
            'commentLength', 'isLatinLanguage', 'positionWithinSession', 'revisionAction', 'revisionPrevAction',
            'revisionSubaction', 'revisionSize', 'bytesIncrease', 'minorRevision', 'parentRevisionInCorpus',
            'param1', 'param3', 'param4', 'timeSinceLastRevision', 'numberOfSitelinksAdded', 'numberOfSitelinksRemoved',
            'numberOfSitelinksChanged', 'numberOfLabelsAdded', 'numberOfLabelsRemoved', 'numberOfLabelsChanged',
            'numberOfDescriptionsAdded', 'numberOfDescriptionsRemoved', 'numberOfDescriptionsChanged', 'numberOfAliasesAdded',
            'numberOfAliasesRemoved', 'numberOfClaimsAdded', 'numberOfClaimsRemoved', 'numberOfClaimsChanged', 'numberOfIdentifiersChanged',
            'englishLabelTouched', 'numberOfSourcesAdded', 'numberOfSourcesRemoved', 'numberOfQualifiersAdded', 'numberOfQualifiersRemoved',
            'numberOfBadgesAdded', 'numberOfBadgesRemoved', 'hasP21Changed', 'hasP27Changed', 'hasP54Changed', 'hasP18Changed',
            'hasP569Changed', 'hasP109Changed', 'hasP373Changed', 'hasP856Changed'
            #, 'rollbackReverted', 'undoRestoreReverted'
            ]
'''
use_cols = ['revisionId', 'userId', 'userName', 'groupId', 'timestamp', 'revisionLanguage',
            'alphanumericRatio', 'asciiRatio', 'bracketRatio', 'digitRatio', 'latinRatio',
            'longestCharacterSequence', 'lowerCaseRatio', 'nonLatinRatio', 'punctuationRatio',
            'upperCaseRatio', 'whitespaceRatio', 'arabicRatio', 'bengaliRatio', 'cyrillicRatio',
            'hanRatio', 'brahmiRatio', 'malayalamRatio', 'tamilRatio', 'teluguRatio',
            'badWordRatio', 'containsLanguageWord', 'languageWordRatio', 'lowerCaseWordRatio',
            'upperCaseWordRatio', 'commentCommentSimilarity', 'commentLabelSimilarity', 'commentTailLength',
            'commentSitelinkSimilarity', 'property', 'itemValue', 'literalValue', 'isPrivilegedUser',
            'isRegisteredUser', 'isBotUser', 'isHuman', 'commentLength', 'isLatinLanguage', 'positionWithinSession',
            'revisionAction', 'revisionPrevAction', 'revisionSubaction', 'revisionTag', 'revisionSize',
            'minorRevision', 'parentRevisionInCorpus', 'englishLabelTouched']
            
header_cols = ['revisionId', 'userId', 'userName', 'groupId', 'timestamp', 'revisionLanguage', 'contentType', 'commentTail', 'itemId', 
'englishItemLabel', 'superItemId', 'latestInstanceOfItemId', 'latestEnglishItemLabel', 'alphanumericRatio', 'asciiRatio', 'bracketRatio', 
'digitRatio', 'latinRatio', 'longestCharacterSequence', 'lowerCaseRatio', 'nonLatinRatio', 'punctuationRatio', 'upperCaseRatio', 'whitespaceRatio',
 'arabicRatio', 'bengaliRatio', 'cyrillicRatio', 'hanRatio', 'brahmiRatio', 'malayalamRatio', 'tamilRatio', 'teluguRatio', 'badWordRatio', 
 'containsLanguageWord', 'containsURL', 'languageWordRatio', 'longestWord', 'lowerCaseWordRatio', 'proportionOfLinksAdded', 'proportionOfQidAdded', 
 'upperCaseWordRatio', 'proportionOfLanguageAdded', 'commentCommentSimilarity', 'commentLabelSimilarity', 'commentTailLength', 'commentSitelinkSimilarity', 
 'property', 'itemValue', 'literalValue', 'isPrivilegedUser', 'isRegisteredUser', 'userCity', 'userContinent', 'userCountry', 'userCounty', 'userRegion', 
 'userTimeZone', 'isBotUser', 'hasListLabel', 'isHuman', 'labelCapitalizedWordRatio', 'labelContainsFemaleFirstName', 'labelContainsMaleFirstName', 
 'numberOfLabels', 'numberOfDescriptions', 'numberOfAliases', 'numberOfStatements', 'numberOfSitelinks', 'numberOfQualifiers', 'numberOfReferences', 
 'numberOfBadges', 'isLivingPerson', 'commentLength', 'isLatinLanguage', 'positionWithinSession', 'revisionAction', 'revisionPrevAction', 'revisionSubaction', 
 'revisionTag', 'revisionSize', 'bytesIncrease', 'minorRevision', 'parentRevisionInCorpus', 'param1', 'param3', 'param4', 'timeSinceLastRevision', 
 'numberOfSitelinksAdded', 'numberOfSitelinksRemoved', 'numberOfSitelinksChanged', 'numberOfLabelsAdded', 'numberOfLabelsRemoved', 'numberOfLabelsChanged', 
 'numberOfDescriptionsAdded', 'numberOfDescriptionsRemoved', 'numberOfDescriptionsChanged', 'numberOfAliasesAdded', 'numberOfAliasesRemoved', 'numberOfClaimsAdded', 
 'numberOfClaimsRemoved', 'numberOfClaimsChanged', 'numberOfIdentifiersChanged', 'englishLabelTouched', 'numberOfSourcesAdded', 'numberOfSourcesRemoved', 
 'numberOfQualifiersAdded', 'numberOfQualifiersRemoved', 'numberOfBadgesAdded', 'numberOfBadgesRemoved', 'hasP21Changed', 'hasP27Changed', 'hasP54Changed', 
 'hasP18Changed', 'hasP569Changed', 'hasP109Changed', 'hasP373Changed', 'hasP856Changed', 'rollbackReverted', 'undoRestoreReverted']

encode_data = {}


def get_encode(key,value):
    global encode_data
    try:
        return encode_data[key][value]
    except: 
        cat_dictionary = encode_data.get(key)
        max_label = max(cat_dictionary.values()) + 1
        cat_dictionary[value] = max_label
        encode_data[key] = cat_dictionary
        return max_label

def to_json(x):
    return json_normalize(ujson.loads(x))

def get_feature(x,f):
    try:
        return str(x[f][0])
    except:
        return 'nan'

def delta_time(row):
    try:
        return abs((row['timestamp'] - row['timestampParent']).total_seconds()) / 60.0
    except:
        return -1

def delta_ratio(row, f1, f2):
    try:
        return abs(row[f1] - row[f2])
    except:
        return -1

def process(X, encode_file):
    features = X.columns.tolist()
    if 'undoRestoreReverted' in features:
        del X['undoRestoreReverted']
    if 'rollbackReverted' in features:
        del X['rollbackReverted']
    if 'ROLLBACK_REVERTED' in features:
        del X['ROLLBACK_REVERTED'] #will merge with truth later


    X['timestamp'] = X.timestamp.map(lambda x: datetime.datetime.strptime(x, date_format))
    X['day'] = X.timestamp.map(lambda x: x.day)
    X['month'] = X.timestamp.map(lambda x: x.month)
    X['year'] = X.timestamp.map(lambda x: x.year)
    X['hour'] = X.timestamp.map(lambda x: x.hour)
    X['min'] = X.timestamp.map(lambda x: x.minute)
    X['date_of_week'] = X.timestamp.map(lambda x: x.weekday())
    # lag features
    del X['timestamp']

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = X.select_dtypes(include=numerics).columns.tolist()
    ratio_val_cols = [f for f in X.columns.tolist() if (f.find("Ratio") > -1 or f.find("Value") > -1)]
    X[ratio_val_cols] = X[ratio_val_cols].applymap(lambda x: np.nan if type(x) == str else x)

    # Ratio feature
    X['upperLowerCaseRatio'] = X['upperCaseRatio'] / X['lowerCaseRatio']
    X['latinNonLatinRatio'] = X['latinRatio'] / X['nonLatinRatio']
    X['tailCommentRatio'] = X['commentTailLength'] / X['commentLength']
    ratio_val_cols.append('upperLowerCaseRatio')
    ratio_val_cols.append('latinNonLatinRatio')
    ratio_val_cols.append('tailCommentRatio')
    features_numeric.extend(ratio_val_cols)

    boolean_cols = [f for f in X.columns.tolist() if (f.startswith("is") or f.startswith("has") or f.lower().find("contains") > -1 )]
    boolean_cols.append('parentRevisionInCorpus') 
    boolean_cols.append('englishLabelTouched')   
    boolean_cols.append('minorRevision') 
    X[boolean_cols] = X[boolean_cols].applymap(lambda x: 1 if x=='T' else 0 if x=='F' else -1)
    features_numeric.extend(ratio_val_cols)
    features_numeric.extend(boolean_cols)
    # other feature
    # X['revisonPart'] = X['revisionAction'].map(lambda x: str(x).replace("Creation", "").replace("wbset", ""))
    #X['isTest'] = X['latestEnglishItemLabel'].map(lambda x: 1 if (str(x).lower().find('test') > -1) else 0)
    #X['isDigitInUserName'] = X['userName'].map(lambda x: int(any(i.isdigit() for i in str(x))))

    features_numeric = [f for f in X.columns.tolist() if f not in cat_features]

    # na
    for f in features_numeric:
        try:
            X[f] = X[f].astype(float).fillna(-1)
        except:
            print 'Error at... ' + f

    X.fillna('missing', inplace=True)
    '''
    if not os.path.exists(encode_file):
        X.drop(cat_features, inplace=True, axis=1)
    else:
        global encode_data
        encode_data = pickle.load(open(encode_file,'rb'))
        for cat in cat_features:
            X[cat] = X[cat].map(lambda x: get_encode(cat,x))'''

    for cat in cat_features:
       # try:
        print cat
        encode_data = pickle.load(open(encode_file + cat + ".pickle",'rb'))
        X[cat] = encode_data.transform(X[cat])
        #except:
        #del X[cat]

    print "# Numeric features: " + str(len(features_numeric))
    
    
    print X.head(5)
    features = X.columns.tolist()
    print "# Total features: " + str(len(features))
    print features
    return X, features
