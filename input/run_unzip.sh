#unzip feature from java
bzip2 -d wdvc16_2012_10.features.bz2
bzip2 -d wdvc16_2012_11.features.bz2
bzip2 -d wdvc16_2013_01.features.bz2
bzip2 -d wdvc16_2013_03.features.bz2
bzip2 -d wdvc16_2013_05.features.bz2
bzip2 -d wdvc16_2013_07.features.bz2
bzip2 -d wdvc16_2013_09.features.bz2
bzip2 -d wdvc16_2013_11.features.bz2
bzip2 -d wdvc16_2014_01.features.bz2
bzip2 -d wdvc16_2014_03.features.bz2
bzip2 -d wdvc16_2014_05.features.bz2
bzip2 -d wdvc16_2014_07.features.bz2
bzip2 -d wdvc16_2014_09.features.bz2
bzip2 -d wdvc16_2014_11.features.bz2
bzip2 -d wdvc16_2015_01.features.bz2
bzip2 -d wdvc16_2015_03.features.bz2
bzip2 -d wdvc16_2015_05.features.bz2
bzip2 -d wdvc16_2015_07.features.bz2
bzip2 -d wdvc16_2015_09.features.bz2
bzip2 -d wdvc16_2015_11.features.bz2
bzip2 -d wdvc16_2016_01.features.bz2

cat wdvc16_*.features | awk -F, '($1!="revisionId")' > features/train_feature_all.csv
rm -rf wdvc16_*.features
cd ./features/
split -l 2500000 train_feature_all.csv
mv xaa chunk1.csv
mv xab chunk2.csv
mv xac chunk3.csv
mv xad chunk4.csv
mv xae chunk5.csv
mv xaf chunk6.csv
mv xag chunk7.csv
mv xah chunk8.csv
mv xai chunk9.csv
mv xaj chunk10.csv
mv xak chunk11.csv
mv xal chunk12.csv
mv xam chunk13.csv
mv xan chunk14.csv
mv xao chunk15.csv
mv xap chunk16.csv
mv xaq chunk17.csv
mv xar chunk18.csv
mv xas chunk19.csv
mv xat chunk20.csv
mv xau chunk21.csv
mv xav chunk22.csv
mv xaw chunk23.csv
mv xax chunk24.csv
mv xay chunk25.csv
mv xaz chunk26.csv
