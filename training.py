import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

def load_intents(file_path):
    """Load intents data from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def preprocess_data(data):
    """Preprocess the intent data."""
    sentences = []
    labels = []
    response_list = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            sentences.append(pattern)
            labels.append(intent['tag'])
        response_list.append(intent['responses'])

    return sentences, labels, response_list

def encode_labels(labels):
    """Encode text labels into numerical values."""
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(labels)
    return lbl_encoder.transform(labels), lbl_encoder

def prepare_tokenizer(sentences, vocab_size=1000, oov_token="<OOV>"):
    """Create and fit the tokenizer."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def prepare_sequences(tokenizer, sentences, max_len=20):
    """Convert sentences into padded sequences."""
    sequences = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(sequences, truncating='post', maxlen=max_len)

def build_model(vocab_size, embedding_dim, max_len, num_classes):
    """Build and compile the neural network model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def save_model(model, file_path):
    """Save the trained model."""
    model.save(file_path)

def save_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    # Load and preprocess data
    data = load_intents('intents.json')
    sentences, labels, _ = preprocess_data(data)

    # Encode labels
    encoded_labels, lbl_encoder = encode_labels(labels)

    # Prepare tokenizer and sequences
    tokenizer = prepare_tokenizer(sentences)
    padded_sequences = prepare_sequences(tokenizer, sentences)

    # Define model parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 16
    max_len = 20
    num_classes = len(set(labels))  # Unique tags

    # Build and train the model
    model = build_model(vocab_size, embedding_dim, max_len, num_classes)
    model.summary()
    model.fit(padded_sequences, np.array(encoded_labels), epochs=500)

    # Save the model and preprocessors
    save_model(model, 'chatbot.keras')
    save_pickle(tokenizer, 'tokenizer.pickle')
    save_pickle(lbl_encoder, 'label_encoder.pickle')

if __name__ == "__main__":
    main()
