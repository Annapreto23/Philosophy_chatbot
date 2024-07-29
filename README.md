# Philosophy Chatbot

This project is a simple philosophy chatbot that provides quotes based on different themes. The chatbot uses a machine learning model to classify user input into predefined categories and responds with relevant philosophical quotes.

## Overview

The philosophy chatbot allows users to interact with it by asking for quotes on various philosophical themes. It uses a pre-trained model to classify user input and deliver quotes accordingly. You can also train the model with new intents using the provided script.

## Features

- **Quote Generation**: Provides quotes from various philosophical themes and authors.
- **User Interaction**: Simple command-line interface for interacting with the chatbot.
- **Model Training**: Train the chatbot on new intents using the `training.py` script.

## Setup

To set up and use the chatbot, follow these steps:

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/philosophy-chatbot.git
   cd philosophy-chatbot
   ```

2. **Install Dependencies**

   Install the required Python packages using `pip`. Ensure you have Python 3.6+.

   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare Data**

   Ensure you have the `intents.json` file with the correct format for intents and responses.

4. **Train the Model**

   Use the `training.py` script to train the model on the data provided in `intents.json`. This script will generate the necessary model, tokenizer, and label encoder files.

   ```sh
   python training.py
   ```

5. **Run the Chatbot**

   Once the model is trained, you can start the chatbot using:

   ```sh
   python chatbot.py
   ```

## File Structure

- `chatbot.py`: Main script to run the chatbot. It loads the trained model and interacts with the user.
- `training.py`: Script to train the model using the data from `intents.json`. It saves the model, tokenizer, and label encoder.
- `intents.json`: File containing the philosophical themes and quotes.
- `chatbot.keras`: Saved Keras model file (generated by `training.py`).
- `tokenizer.pickle`: Saved tokenizer object (generated by `training.py`).
- `label_encoder.pickle`: Saved label encoder object (generated by `training.py`).
- `requirements.txt`: List of Python dependencies needed for the project.

## Usage

1. **Train the Model**

   Before running the chatbot, ensure the model is trained:

   ```sh
   python training.py
   ```

2. **Run the Chatbot**

   Start the chatbot and interact with it through the command line:

   ```sh
   python chatbot.py
   ```

   To exit the chat, type `quit`.

   ```sh
   Start messaging with the bot (type 'quit' to stop)!
   User: Tell me a quote about love
   ChatBot: "The only thing we have to fear is fear itself." - Franklin D. Roosevelt
   ```


# Philosophy_chatbot
# Philosophy_chatbot
# Philosophy_chatbot
# Philosophy_chatbot
# Philosophy_chatbot
# Philosophy_chatbot
