import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# import nltk

def lemmatize_text(text):
    wl = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmas = [wl.lemmatize(token) for token in tokens if token.isalpha()]
    return " ".join(lemmas)

if __name__ == "__main__":
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    data = {
        'text' : [
            'Cats are running quickly.',
            'Dogs are barking loudly.',
            'He studies in university.',
            'They are enjoying the holidays'
        ]
    }

    df = pd.DataFrame(data)
    print(f"Original Data\n {df}")
    df['tokens'] = df['text'].apply(lambda x: word_tokenize(x))
    df['lemmatized_text'] = df['text'].apply(lemmatize_text)
    print(f"After lemmatization \n{df}")
