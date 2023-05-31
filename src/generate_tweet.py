import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


# Load the saved tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the saved model
model = tf.keras.models.load_model("models/donald_model.h5")

# Define the maximum sequence length
max_sequence_len = model.input_shape[1]

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)

        # Sample the next word based on predicted probabilities
        predicted_probs = predicted_probs.ravel()  
        predicted_class = np.random.choice(np.arange(len(predicted_probs)), p=predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_class:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()

# Get user input
seed_input = input("Enter a text: ")
num_words = int(input("Enter the number of words to generate: "))

# Generate text
generated_text = generate_text(seed_input, num_words)
print("Generated Text:")
print(generated_text)
