

import streamlit as st
import pandas as pd
#import plotly.figure_factory as ff
import tweepy
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from st_aggrid import AgGrid, GridOptionsBuilder
#import altair as alt


with st.container():
	
	# TITULO
	st.title('Twitter prueba')

	col1, col2 = st.columns([2, 3])
	selected_rows = []

	with col1:
		# HEADER COL 1
		st.subheader("LISTADO DE TT")
		df = pd.DataFrame({"hashtag": ["#love", "#ukranie", "#madrid", "#dog", "#anathebest"]})
		options_builder = GridOptionsBuilder.from_dataframe(df)
		options_builder.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
		options_builder.configure_column("hashtag", editable=False) 
		options_builder.configure_pagination(paginationAutoPageSize=True)
		options_builder.configure_selection("single", use_checkbox=True)
		
		grid_options = options_builder.build()

		grid_return = AgGrid(
			df, 
			grid_options,
			width='100%',			
		) 
		
		df = grid_return['data']
		selected_rows = grid_return["selected_rows"]
		selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')
		st.write(selected_df)


	with col2:
		# DESCRIPCIÓN
		#st.write("PRUEBA PARA TFM TWITTER")
		# FORMULARIO HASHTAG
		if len(selected_rows) == 0:
			hashtag = st.text_input('Introduce un hashtag', "#love")
		else:
			#hashtag = st.text_input('Introduce un hashtag', getattr(selected_rows[0],'hashtag'))
			hashtag = st.text_input('Introduce un hashtag', selected_rows[0]["hashtag"])

# Model importing
# Load the model (only executed once!)
# NOTE: Don't set ttl or max_entries in this case

@st.cache
def load_model():
	return AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
  
#@st.cache
#def load_tokenizer():
#	return AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
  
model = load_model()
#tokenizer = load_tokenizer()

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
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

#labels = ['Positive', 'Neutral', 'Negative']
#sections = [sentimientos.count('Positive'), sentimientos.count('Neutral'), sentimientos.count('Negative')]
#colors = ['g', 'y', 'r']
#chart_data = pd.DataFrame([sections], columns = labels)
#st.bar_chart(chart_data)

with col2:
	labels = 'Positive', 'Neutral', 'Negative'
	sections = [sentimientos.count('Positive'), sentimientos.count('Neutral'), sentimientos.count('Negative')]
	colors = ['g', 'y', 'r']
	fig1, ax1 = plt.subplots()
	ax1.pie(sections, labels=labels, colors=colors,
		startangle=90,
		explode = (0, 0, 0),
		autopct = '%1.2f%%')
	ax1.axis('equal') # Try commenting this out.
	st.write("Pie Chart Twitter Sentiment Example")
	st.pyplot(fig1)


