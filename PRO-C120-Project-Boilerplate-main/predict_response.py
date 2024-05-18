#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# words to be igonred/omitted while framing the dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

# load the model
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    bag=[]
    bag_of_words = []

    for intent in intents['intents']:

        # Add all patterns and tags to a list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                        
            user_input.append((pattern_word, intent['tag']))
              
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
    stem_words = get_stem_words(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words, classes, user_input
    
    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
   # extract the class from the predicted_class_label
   predicted_class = "user_input"

   # now we have the predicted tag, select a random response

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # choose a random bot response
        bot_response = "Hi there!"
    
        return bot_response
    

print("Hi I am Stella, How Can I help you?")

while True:

    # take input from the user
    user_input = input('Type you message here : ')

    response = bot_response(user_input)
    print("Bot Response: ", response)