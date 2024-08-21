import pandas as pd

# Load the dataset
url = 'https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/download'
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())
# Check for missing values
print(data.isnull().sum())

# Fill missing values if necessary
data.fillna('', inplace=True)

# Example preprocessing: Convert text to lowercase
data['text'] = data['text'].str.lower()
from textblob import TextBlob

# Define a function to get sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply the function to the text column
data['sentiment'] = data['text'].apply(get_sentiment)
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(data['sentiment'], bins=30, kde=True)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
# Assuming there's a 'brand' column
plt.figure(figsize=(12, 8))
sns.boxplot(x='brand', y='sentiment', data=data)
plt.title('Sentiment Scores by Brand')
plt.xticks(rotation=45)
plt.xlabel('Brand')
plt.ylabel('Sentiment Score')
plt.show()
plt.savefig('sentiment_distribution.png')
