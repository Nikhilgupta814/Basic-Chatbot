import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import joblib
import colorama
from colorama import Fore, Style, Back
import random


with open("intents.json") as file:
    data = json.load(file)

def chat():
    model = keras.models.load_model('chat_model')
    tokenizer = joblib.load('tokenizer.pkl')
    lbl_encoder = joblib.load('lbl_encoder.pkl')

    max_len = 20
    
    while True:
        print(Fore.RED + "User: " + Style.RESET_ALL, end= "")
        inp = input()
        if inp.lower()=="quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating = 'post',
         maxlen = max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag']==tag:
                print(Fore.GREEN + "ChatBot: " + Style.RESET_ALL, np.random.choice(i['responses']))

print(Fore.BLACK + "Start Chating With Me (Type quit to stop)" + Style.RESET_ALL)
chat()