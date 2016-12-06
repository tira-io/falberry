# falberry
The Falberry Vandalism Detector

# Required Data
- Download 21 zipped file from http://www.wsdm-cup-2017.org/vandalism-detection.html and store ./input folder

# Requirements
- Java 8
	+ Jar file built from https://github.com/Zepx/wsdmcup17-wdvd-baseline-feature-extraction
- Python 2.7
    + xgboost
    + pandas
    + sklearn
    
# Step to train
1- Feature extraction: ./input/run_feature.sh
2- Unzip feature and split data: ./input/run_unzip.sh
3- Prepare categorical feature to encode: pypy ./src/encode_v0.py
4- Encode categorical feature: python ./src/encode_v1.py
5- Train model: ./src/train_chunks.sh

# Step to predict
1- python xgb_predictior.py
