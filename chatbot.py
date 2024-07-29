import json
import numpy as np
import pickle
import random
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
from colorama import Fore, Style

# Initialize colorama for terminal colors
colorama.init()

# Constants
MAX_LEN = 50
MODEL_PATH = 'chatbot.keras'
TOKENIZER_PATH = 'tokenizer.pickle'
ENCODER_PATH = 'label_encoder.pickle'
INTENTS_PATH = 'intents.json'

def load_data():
    """Load the data and required objects from files."""
    try:
        with open(INTENTS_PATH) as file:
            data = json.load(file)
    except FileNotFoundError:
        print(Fore.RED + "Error: The intents file was not found." + Style.RESET_ALL)
        exit()
    return data

def load_model_and_objects():
    """Load the model, tokenizer, and label encoder."""
    try:
        model = keras.models.load_model(MODEL_PATH)
    except IOError:
        print(Fore.RED + "Error: The model file was not found or cannot be loaded." + Style.RESET_ALL)
        exit()

    try:
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except IOError:
        print(Fore.RED + "Error: The tokenizer file was not found or cannot be loaded." + Style.RESET_ALL)
        exit()

    try:
        with open(ENCODER_PATH, 'rb') as enc:
            lbl_encoder = pickle.load(enc)
    except IOError:
        print(Fore.RED + "Error: The label encoder file was not found or cannot be loaded." + Style.RESET_ALL)
        exit()

    return model, tokenizer, lbl_encoder

def predict_tag(model, tokenizer, lbl_encoder, user_input):
    """Predict the tag for the user's input."""
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', maxlen=MAX_LEN)
    result = model.predict(padded_sequences)
    tag_index = np.argmax(result)
    tag = lbl_encoder.inverse_transform([tag_index])[0]
    #print(tag)
    return tag

def get_response(tag, data):
    """Get a random response for a given tag."""
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

def chat():
    """Manage the conversation loop with the user."""
    data = load_data()
    model, tokenizer, lbl_encoder = load_model_and_objects()
    
    print(Fore.RED + "Start messaging with the bot (type 'quit' to stop)!" + Style.RESET_ALL)
    
    while True:
        user_input = input(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL)
        if user_input.lower() == "quit":
            print(Fore.BLUE + "Exiting chat. Have a great day!" + Style.RESET_ALL)
            break
        
        tag = predict_tag(model, tokenizer, lbl_encoder, user_input)
        response = get_response(tag, data)
        print(Fore.BLUE + "ChatBot:" + Style.RESET_ALL, response)

if __name__ == "__main__":
    chat()
