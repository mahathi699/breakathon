import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk

# Read in data
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Sentimental Analysis_Amazon Reviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)

ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()
example = df['Text'][50]
print(example)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

import nltk
example = "370 chars 'Got a free package …'"
tokens = nltk.word_tokenize(example)
tokens[:10]

tagged = nltk.pos_tag(tokens)
tagged[:10]

nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()

sia.polarity_scores('I am so happy!')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)

# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders.merge(df, how='left')
vaders.head()
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

# Pre-Trained Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# VADER results on example
example = "70 chars 'Got a free package ..."
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores(example))

# Run for RoBERTa Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2],
}
print(scores_dict)

# Define a function to calculate sentiment scores using RoBERTa
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2],
    }
    return scores_dict

res = polarity_scores_roberta(example)
print(res)

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f"Broke for id {myid}")

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# Compare Scores between Models
results_df.columns

# Combine and Compare
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos', 
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='tab10')
plt.show()

# Review Examples: Positive 1-Star and Negative 5-Star Reviews
# Let's look at some examples where the model scoring and reviews differ

# Example for 1-Star Reviews sorted by RoBERTa Positive score
results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]

# Example for 1-Star Reviews sorted by RoBERTa Negative score
results_df.query('Score == 1') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]

# Example for 5-Star Reviews sorted by Vader Negative score
results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]

# Example for 5-Star Reviews sorted by RoBERTa Negative score
results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]

# Using Hugging Face pipeline for sentiment analysis
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

# Examples with sentiment pipeline
sent_pipeline("I love sentiment analysis!")
# Output: [{'label': 'POSITIVE', 'score': 0.9997858363741643}]

sent_pipeline("Make sure to like and subscribe!")
# Output: [{'label': 'POSITIVE', 'score': 0.9991742968559265}]

sent_pipeline("Booo")
# Output: [{'label': 'NEGATIVE', 'score': 0.9936267173572466}]
