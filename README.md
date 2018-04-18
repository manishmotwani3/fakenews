# fakenews
Dataset 1: FactCheck dataset :  https://homes.cs.washington.edu/~hrashkin/factcheck.html

Dataset 2: Kaggle dataset(Fake, 1 year old) + Signal Media Dataset(Real, Sample 50000 news articles) : http://research.signalmedia.co/newsir16/signal-dataset.html, https://www.kaggle.com/mrisdal/fake-news 

Dataset:

Model 1 trained on Dataset1(4 way classification)

Model 1 test on Dataset2 (2 way classification)

Model 2 : (Model 1 + additional features from Dataset 2) : Train and test on Dataset 2

Model 3 : (Model 2 - additional features from Dataset 2) : Test on Dataset 1

Analysis:
Which features(from either of the datasets) can contribute to fake news detection?

 
