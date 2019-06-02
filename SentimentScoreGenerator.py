import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# first read your tweet data csv in as a pandas dataframe
df = pd.read_csv("/outpath/", encoding='utf-8', sep = ',')

# define a function analyze_sentiment for your dataframe
def analyze_sentiment(df):
   #initialize empty list
   sentiments = []
   #initialize sentiment analyzer
   sid = SentimentIntensityAnalyzer()

   # df.shape[0] will count the number of rows. FYI df.shape[1] will count the number of columns.
   for i in range(df.shape[0]):
       # score each line of text with polarity scores
       line = df['text'].iloc[i]
       sentiment = sid.polarity_scores(line)
       # append sentiment scores for each line to new columns that are labeled according to the sentiment score labels from vader
       sentiments.append([sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound']])
   # add this data to the data frame
   df[['neg', 'pos', 'neu', 'compound']] = pd.DataFrame(sentiments)
   #give each tweet an overall positive/negative label based on compound score thresholds
   df['Negative'] = df['compound'] < -0.1
   df['Positive'] = df['compound'] > 0.1
   return df

#run your sentiment analysis function on your dataframe.
#return the dataframe with the added columns with sentiment scores & labels

analyze_sentiment(df)

#to save data as a csv, uncomment the following. then you can import your csv to Tableau or other visualization software.
analyze_sentiment(df).to_csv(/document/)

