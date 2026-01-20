import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')

data = {
    'text' : [
        'cats are running quickly',
        'dogs are barking loudly',
        'he studies in university',
        'they are enjoying the holidays'
    ]
}
df = pd.DataFrame(data)
print(f"Original Data {df}")

def stemming(sen):
    tokens = word_tokenize(sen)
    stemm = PorterStemmer()
    stem_words = [stemm.stem(token) for token in tokens]
    return " ".join(stem_words)

df['tokens'] = df['text'].apply(lambda x: word_tokenize(x))
df['stemmed_text'] = df['text'].apply(stemming)
print(f"After stemming \n{df}")

