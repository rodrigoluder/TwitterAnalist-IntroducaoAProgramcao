import tweepy
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

with open('./twitter_keys.txt', 'r') as f:
    api_key = f.readline().strip('\n')
    api_key_secret = f.readline().strip('\n')
    acess_token = f.readline().strip('\n')
    acess_token_secret = f.readline().strip('\n')


auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(acess_token, acess_token_secret)

api = tweepy.API(auth)

palavras_chave_hashtags = input(
    '''Insira aqui as palavras chaves e/ou hashtags relacionadas ao tema de interesse
    separados por espaços:'''
    )

lista_query = []
for palavra in palavras_chave_hashtags.split():
    query_string = palavra + ' -filter:retweets'
    lista_query.append(query_string)


dict_features_ = {}
n = 0

limite = 1_000
qtd_por_palavra = limite/len(lista_query)

for query_string in lista_query:

    print(f'Coletando tweets com o termo: {query_string.split()[0]}\n')

    tweets = tweepy.Cursor(api.search_tweets, q=query_string).items(qtd_por_palavra)
    
    for twt in tqdm(tweets, total=int(qtd_por_palavra)):
        d = twt._json
        dict_features_[n] = {}

        dict_features_[n]['created_at'] = d['created_at']
        dict_features_[n]['hashtags'] = d['entities']['hashtags']#['text']
        dict_features_[n]['text'] = d['text']
        dict_features_[n]['tweet_id'] = d['id']
        dict_features_[n]['user_description'] = d['user']['description']
        dict_features_[n]['user_id'] = d['user']['id']
        dict_features_[n]['user_name'] = d['user']['name']
        dict_features_[n]['user_@'] = d['user']['screen_name']


        dict_features_[n]['reply_tweet_id'] = d['in_reply_to_status_id']
        dict_features_[n]['reply_user_@'] = d['in_reply_to_screen_name']
        dict_features_[n]['reply_user_id'] = d['in_reply_to_user_id']


# Salvando dataframe com tweets 
pd.DataFrame.from_dict(dict_features_, orient='index').to_csv(f'./df_tweets.csv', index=False)