import pandas as pd
from collections import Counter

# importando dados pre-processados e limpos

df_tweets = pd.read_csv(f'./df_tweets_limpos.csv', index=False)


tweets_hashtags = []
username_hashtags = []
descritpions_hashtags = []

for hstg in df_tweets['hashtags_list'].dropna():
    if type(hstg) == list:
        tweets_hashtags.extend(hstg)
    else:
        tweets_hashtags.append(hstg)

df_user_info = df_tweets.copy()
for hstg_d in df_user_info['descriptions_hashtags'].dropna():
    if type(hstg_d) == list:
        descritpions_hashtags.extend(hstg_d)
    else:
        descritpions_hashtags.append(hstg_d)
        
for hstg_u in df_user_info['username_hashtags'].dropna():
    if type(hstg_d) == list:
        username_hashtags.extend(hstg_u)
    else:
        username_hashtags.append(hstg_u)
    

tweets_hashtags_ranked = {k:v for k, v in sorted(Counter(tweets_hashtags).items(), key=lambda x:x[1], reverse=True)}
description_hashtags_ranked = {k:v for k, v in sorted(Counter(descritpions_hashtags).items(), key=lambda x:x[1], reverse=True)}
username_hashtags_ranked = {k:v for k, v in sorted(Counter(username_hashtags).items(), key=lambda x:x[1], reverse=True)}



tweets_htsg = pd.DataFrame.from_dict(tweets_hashtags_ranked, orient='index')

username_htsg = pd.DataFrame.from_dict(username_hashtags_ranked, orient='index')

bios_htsg = pd.DataFrame.from_dict(description_hashtags_ranked, orient='index')

from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngrams(corpus, n=None):
    corpus = [' '.join(tweet) for tweet in corpus]   
    vec = CountVectorizer(ngram_range=(1, 3)).fit(corpus) # de 1 a 3 n-grams 
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return [[n_gram, freq] for n_gram, freq in words_freq][:200] # seleciona as 200 palavras mais frequentes


top_ngrams = pd.DataFrame(get_top_ngrams(df_tweets['prep_text']), columns=['ngram', 'frequencia'])

top_ngrams_desc = pd.DataFrame(get_top_ngrams(df_tweets['pre_description']), columns=['ngram', 'frequencia'])