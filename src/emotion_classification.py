import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import os

# Load the model
model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_classifier = pipeline("sentiment-analysis", model=model_name, return_all_scores=False)

# Load the CSV file containing the tweets into a pandas DataFrame
df = pd.read_csv('data/trumptweets.csv')

# Define the colors for each emotion
emotion_colors = {
    'anger': 'red',
    'joy': 'yellow',
    'sadness': 'gray',
    'fear': 'purple',
    'surprise': 'pink',
    'disgust': 'green',
    'neutral': 'blue'
}

# Classify the emotions for each tweet and store the results in a new column
df['emotion'] = df['content'].apply(lambda x: emotion_classifier(x)[0]['label'])

# Calculate the distribution of emotions
emotion_distribution = df['emotion'].value_counts(normalize=True) * 100

# Create a bar chart to visualize the distribution of emotions
plt.figure(figsize=(10, 6))
emotion_distribution.plot(kind='bar', color=[emotion_colors.get(emotion, 'gray') for emotion in emotion_distribution.index])
plt.xlabel('Emotion')
plt.ylabel('Percentage')
plt.title('Distribution of Emotions in Donald Trump Tweets')
plt.savefig('out/emotions.png')

# Save the emotions and their labels to a CSV file
emotion_labels = pd.DataFrame({'emotion': emotion_distribution.index, 'percentage': emotion_distribution.values})
emotion_labels.to_csv('out/emotion_labels.csv', index=False)
