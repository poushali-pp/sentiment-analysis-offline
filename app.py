import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample tweet data
data = {
    'Tweet': [
        "I love the new design of the iPhone!",
        "This is the worst movie I've seen this year.",
        "I feel indifferent about the new policy.",
        "Great customer service at the store today.",
        "The weather is terrible and ruining my plans.",
        "Can't wait for the weekend!",
        "Traffic today was horrible.",
        "Amazing performance by the team!",
        "Nothing exciting happened today.",
        "Feeling a bit tired but okay."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Tweet'].apply(get_sentiment)

# Print results
print(df)

# Optional: Plot sentiment counts
df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

