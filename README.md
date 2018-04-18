# fakenews
Dataset 1: FactCheck dataset :  https://homes.cs.washington.edu/~hrashkin/factcheck.html

Dataset 2: Kaggle dataset(Fake, 1 year old) + Signal Media Dataset(Real, Sample 50000 news articles) : http://research.signalmedia.co/newsir16/signal-dataset.html, https://www.kaggle.com/mrisdal/fake-news 

# RQ: What features of news articles can characterize fake news?

Hypothesis 1: Whats the accuracy of a model to predict if the given news is Hoax, Satire, Propaganda, or Trusted News?
Experiment 1:
Model 1 train and test on Dataset1 (4 way classification)
Expected result: Compare the results with factcheck paper results

Hypothesis 2: Whats the accuracy of a model to classify the news as fake or trusted?
Experiment 2.1:
Model 1 test on Dataset2 (2 way classification) (using pre-trained model from hypothesis1)
Experiment 2.2:
Model 1 train and test on Dataset2 (a new model)
Expected results: compare results with existing studies


Hypothesis 3: Whats the contribution of additional features (Heading, source of article(need to get sources for real news), SPAM score, Published Time) in characterizing fake news?
Experiment 3.i:
Model 2 = (Model 1 + additional feature f_i) 
Model 2 train and test on Dataset 2
Expected results: feature i improves accuracy or not

Hypothesis 4: Which features (from either of the datasets) can contribute to fake news detection?
Experiment 3:
Model 3 = (Model 2 - additional features) 
Model 3 test on Dataset1 
Expected results: feature i contributed to fake news detection or not
 
