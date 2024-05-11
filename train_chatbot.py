import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

# Read intents from JSON file
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Process intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add documents to the corpus
        documents.append((word, intent['tag']))
        # Add unique intent tags to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase, and remove duplicates from words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))

# Print information about the corpus
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save processed words and classes using Pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []

# Initialize an empty bag of words for each document
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create bag of words representation
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Create one-hot encoded output
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1

    # Append bag of words and output to training list
    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Convert training list to NumPy array
# Pad sequences to ensure consistent length
max_seq_length = max(len(seq[0]) for seq in training)
training_data = np.array([np.pad(seq[0], (0, max_seq_length - len(seq[0])), mode='constant') for seq in training])
training_labels = np.array([seq[1] for seq in training])

print("Training data created")

# Create neural network model
model = Sequential()
model.add(Dense(128, input_shape=(training_data.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(training_labels.shape[1], activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit and save the model
hist = model.fit(training_data, training_labels, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Model created")
