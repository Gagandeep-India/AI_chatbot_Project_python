import random, json, pickle, numpy as np, nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.load(open('intents.json'))

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('first_chatbot.keras')

# as it again gives everything in 0's and 1's we need it in words so we do the following

def clean_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#checking if the words from the sentence in the
def bag_words(sentence):
    sentence_words = clean_sentences(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict(sentence):
    bow = bag_words(sentence)
    res = model.predict(np.array([bow]))
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res[0]) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True) #sorting in descending so that higher value comes first
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def generate(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ''
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Go the chatbot is running!')
if_condition = ''
while if_condition != 'Terminate':
    message = input('')
    if_condition = 'Terminate'
    ints = predict(message)
    res = generate(ints, intents)
    print(res)

#
