import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the trained model
model = load_model('models/donald_model.h5')

# Maximum sequence length
max_sequence_len = model.input_shape[1]

# Generate text based on user input
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# User input
input_text = input("Enter a seed text: ")
num_words = int(input("Enter the number of words to generate: "))

# Generate text
generated_text = generate_text(input_text, num_words, model, max_sequence_len)
print("Generated text:")
print(generated_text)
