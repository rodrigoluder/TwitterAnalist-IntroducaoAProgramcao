import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re

# pacotes de pre-processamento para nlp (nltk)
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('wordnet')
import string


# importando dataframe gerado por coletando_tweets.py

df_tweets = pd.read_csv('./df_tweets.csv')
df_tweets = df_tweets[['text', 'hashtags', 'user_description', 'user_name']] # selecionando principais features


# funcoes auxiliares pra coletar hashtags do texto, bio e nome

HASHTAG = re.compile(r'#(\w+)')

def hashtags_to_list(x):
    try:
      if x == []:
        x = np.NaN
      else:
        x = [x[i]['text'] for i in range(len(x))]
    except:
      x = np.NaN
     
    return x

def get_descriptions_hashtags(d):
  if type(d) != str:
    return np.nan
  else:
    list_hashtags = HASHTAG.findall(d)
    if len(list_hashtags) > 0:
      return list_hashtags
    else:
      return np.nan


def get_username_hashtags(u):
  if type(u) != str:
    return np.nan
  else:
    list_hashtags = HASHTAG.findall(u)
    if len(list_hashtags) > 0:
      return list_hashtags
    else:
      return np.nan


# aplicando as funcoes de hashtags nos dataframes com a abodargem funcional (apply + lambda)

df_tweets['hashtags_list'] = df_tweets.apply(lambda x: hashtags_to_list(x['hashtags']), axis=1)
df_tweets['descriptions_hashtags'] = df_tweets.apply(lambda x: get_descriptions_hashtags(x['user_description']), axis=1)
df_tweets['username_hashtags'] = df_tweets.apply(lambda x: get_username_hashtags(x['user_name']), axis=1)




''''
Importante: Como não domino inteiramente as expressões regulares, algumas dessas expressões são baseadas em códigos de terceiros, 
coletados em foruns como Stack Overflow. Mas, friso, me baseiei somente na parte de regex, que facilita bastante uma busca otimizada em strings. 
A Classe de pre-processamento, no entanto, é de minha autoria.

'''

class Preprocess():
    def __init__(self, lower=True): # lower=True para tornar tokens em letras minusculas
        self.STOPWORDS = set(stopwords.words('portuguese')) # as palavras mais comuns da língua - "eu, a, o, aquele, eles, voce, da ...
        self.pontuations = string.punctuation # lista de pontuações
        self.tokenizer = RegexpTokenizer(r'\w+') # funcao pra tokenizar os textos
        self.stemmer = nltk.PorterStemmer() #funcao pra stemmatizar tokens
        self.lemmatizer = nltk.WordNetLemmatizer() # funcao pra lemmatizar tokens
        self.hashtags = re.compile(r'#[-_.?&~;+=/#0-9A-Za-z]{1,2076}') # regex pra remover hashtags
        self.mentions = re.compile(r'@[-_.?&~;+=/#0-9A-Za-z]{1,2076}') # regex pra remover mentions em tweets
        self.lower = lower


    # removendo stopwords
    # retorna a lista de palavras que nao estão no conjunto de stopwords, ignorando-as
    def cleaning_stopwords(self, text, STOPWORDS):
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

    # removendo puntuacoes
    def cleaning_punctuations(self, text, pontuacoes):
        translator = str.maketrans('', '', pontuacoes)
        return text.translate(translator)

    # removendo caracteres repetidos
    def cleaning_repeating_char(self, text):
        return re.sub(r'(.)\1+', r'\1', text)

    # removendo email
    def cleaning_email(self, text):
        return re.sub('@[^\s]+', ' ', text)

    # removendo URL's
    def cleaning_URLs(self, text):
        return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)

    # removendo numeros
    def cleaning_numbers(self, text):
        return re.sub('[0-9]+', '', text)

    # Stemming -  reduzir palavras flexionadas
    def stemming_on_text(self, data, stm):
        text = [stm.stem(word) for word in data]
        return data

    #Lemmatizer - agrupar as formas flexionadas de uma palavra para uma unica feature
    def lemmatizer_on_text(self, data, lm):
        text = [lm.lemmatize(word) for word in data]
        return data
                
    # transformando tweet orignal em tweet pre-processado - limpo
    def transform(self, texts):
        #texts: lista de tweets
        tweets_preprocessados = []
        for t in tqdm(range(len(texts)), total=len(texts), position=0):

            if self.lower:
              tweet = texts[t].lower()
            else:
              tweet = texts[t]
            tweet = self.cleaning_stopwords(tweet, self.STOPWORDS)
            tweet = self.cleaning_punctuations(tweet, self.pontuations)
            tweet = self.cleaning_repeating_char(tweet)
            tweet = self.cleaning_email(tweet)
            tweet = self.cleaning_URLs(tweet)
            tweet = self.cleaning_numbers(tweet)
            tweet = self.hashtags.sub(" ", tweet)
            tweet = self.mentions.sub(" ", tweet)
            tweet = self.tokenizer.tokenize(tweet)
            tweet = self.stemming_on_text(tweet, self.stemmer)
            tweet = self.lemmatizer_on_text(tweet, self.lemmatizer)

            tweets_preprocessados.append(tweet)

        return tweets_preprocessados


prep = Preprocess()
df_tweets['prep_text'] = prep.transform(df_tweets['text'].to_list())
df_tweets['pre_description'] = prep.transform(df_tweets['user_description'].to_list())



# Salvando dataframe com tweets pre-processados e limpos
df_tweets.to_csv(f'./df_tweets_limpos.csv', index=False)
