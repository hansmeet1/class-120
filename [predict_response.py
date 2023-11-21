import nltk,json,pickle,random
import numpy as np
 
ignore_words=["?","!",",",".","'s","'m"]
import tensorflow
from data_preprocessing import get_stem_words
model=tensorflow.keras.models.load_model("./chatbot_model.h5")
intents=json.loads(open("./intents.json").read())
words=pickle.load(open("./words.pk1","rb"))
classes=pickle.load(open("./classes.pk1","rb"))

def preprocess_user_input(user_input):
    token_1=nltk.word_tokenize(user_input)
    token_2=get_stem_words(token_1,ignore_words)
    token_2=sorted(list(set(token_2)))
    bag=[]
    bag_of_words=[]
    for word in words :
        if word in token_2:
            bag_of_words.append(1)
        else :
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_prediction(user_input):
    i=preprocess_user_input(user_input)
    prediction=model.predict(i)
    predicted_label=np.argmax(prediction[0])
    return predicted_label()
