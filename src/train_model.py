import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import re

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load the tweets from the CSV file
df = pd.read_csv('data/trumptweets.csv')
tweets = df['content'].tolist()


# Helper function
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.replace("“", "").replace("”", "")
    return txt.lower()

# Helper function
def clean_text(txt):
    # Convert to lowercase
    txt = txt.lower()
    
    # Remove URLs
    txt = re.sub(r"http\S+|www\S+|https\S+", "", txt, flags=re.MULTILINE)
    
    # Remove usernames
    txt = re.sub(r"@\w+", "", txt)
    
    # Remove hashtags
    txt = re.sub(r"#\w+", "", txt)
    
    # Remove numbers
    txt = re.sub(r"\d+", "", txt)
    
    # Remove punctuation
    txt = "".join(v for v in txt if v not in string.punctuation)
    
    # Remove extra whitespaces
    txt = re.sub(r"\s+", " ", txt)
    
    return txt.strip()


# Clean and lowercase the tweets
tweets = [clean_text(tweet) for tweet in tweets]

# Tokenize the training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
total_words = len(tokenizer.word_index) + 1

# Save the tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Preprocess the training data into sequences and targets
input_sequences = []
target_sequences = []
for line in tweets:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence[:-1])
        target_sequences.append(n_gram_sequence[-1])

# Pad sequences for consistent input shape
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
target_sequences = np.array(target_sequences)

# et batch sizeS
batch_size = 512

# Create the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(max_sequence_len, 1))) 
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')

# Train the LSTM model
model.fit(input_sequences, target_sequences, batch_size=batch_size, epochs=50)

# Save the trained model
model.save('models/donald_model.h5')
