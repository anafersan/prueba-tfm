

import streamlit as st
import pandas as pd
#import plotly.figure_factory as ff
import tweepy
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import altair as alt

# TITULO
st.title('Twitter prueba')
# DESCRIPCIÓN
st.write("PRUEBA PARA TFM TWITTER")
# FORMULARIO HASHTAG
hashtag = st.text_input('Introduce un hashtag', '#Love')

# Model importing
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Twitter API client
bearer_token = "AAAAAAAAAAAAAAAAAAAAAGcdZgEAAAAADAbPiGFS3jT3w7bTECCewFv0oR8%3D4ZAl4MWV5Z8RmSrM99HaA5xhij9UkDxlYugx4tv8YO9Z8PcU0v"
consumer_key = "3UkoMlYsWmPeUDYyTpDHb0V62"
consumer_secret = "n9hqzC2j8KPuqQKwa6hJkM3eBPvN1fwE2Gahr1riygX1LkahUq"
access_token = "1496230187687624709-t1AcjFLOJfRk0zbvKkqYtqRv7uuLiD"
access_token_secret = "NwnmURf8nmm4VnoDVLZwPqxPwXIRmamf5RgtYY4aXmmjn"

client = tweepy.Client(bearer_token, 
                       consumer_key, 
                       consumer_secret, 
                       access_token, 
                       access_token_secret)

#Tweets extraction
query =  hashtag + " lang:en -is:retweet" 

response = client.search_recent_tweets(query=query,
                                     tweet_fields = ["created_at", "text", "source", "public_metrics", "entities"],
                                     user_fields = ["name", "username", "location", "verified", "description", "public_metrics"],
                                     max_results = 100,
                                     expansions='author_id'
                                     )

#Creamos lista vacía
sentimientos = []

#Obtenemos sentimientos de los tweets y almacenamos en la lista
for tweet in response.data:
  texto = tweet.text
  sentiment = sentiment_task(texto)[0]
  label = sentiment['label']
  #print(label)
  sentimientos[len(sentimientos):] = [label]


# Gráfico distribución de sentimientos

labels = ['Positive', 'Neutral', 'Negative']
sections = [sentimientos.count('Positive'), sentimientos.count('Neutral'), sentimientos.count('Negative')]
colors = ['g', 'y', 'r']
chart_data = pd.DataFrame([sections], columns = labels)
st.bar_chart(chart_data)


