import pandas as pd
from nltk.tokenize import word_tokenize
# import nltk

# download model for tokenization
# run once to perform tokenization
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
print(f"Original Data \n{df}")
df['tokens'] = df['text'].apply(lambda x: word_tokenize(x.lower()))
print(f"After tokenization \n{df}")


