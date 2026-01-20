import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
    tfidf = TfidfVectorizer()
    values = tfidf.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(values.toarray(),columns=tfidf.get_feature_names_out())
    print(f"Original Data\n{df}")
    print(f"After tf idf vectorization \n{tfidf_df}")