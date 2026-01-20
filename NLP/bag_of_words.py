import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    data = {
         'text' : [
            'Cats are running quickly.',
            'Dogs are barking loudly.',
            'He studies in university.',
            'They are enjoying the holidays'
        ]
    }
    df = pd.DataFrame(data)
    print(f"Original data \n {df}")
    vectorizer = CountVectorizer()
    values = vectorizer.fit_transform(df['text'])
    bag_of_words = pd.DataFrame(values.toarray(),columns=vectorizer.get_feature_names_out())
    print(f"Bag of words \n {bag_of_words}")