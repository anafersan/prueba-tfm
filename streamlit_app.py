import streamlit as st
import pandas as pd
#import plotly.figure_factory as ff
import tweepy
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import yweather
import json
from datetime import datetime



# PAGE CONFIG 
st.set_page_config(layout="wide")
st.title('Encuentra Treding Topics de Twitter según ubicación')
st.markdown(
        "**Busca la ubicación** sobre la que desees conocer qué temas son tendencia. "
    )
st.markdown(
	"**Selecciona el tema** que más te interese "
    )
st.markdown(
	"**Observa las métricas** calculadas sobre los tweets que se han recogido. "
    )
st.markdown(
        "*Esto es una **demo desarrollada con fines académicos** que en ningún caso debe usarse con fines comerciales* "
    )

@st.cache(allow_output_mutation=True)
def fetch_data(samples):
    dummy_data = {
        "hashtag": ["#love", "#ukranie", "#USA", "#Madrid"],
    }
    return pd.DataFrame(dummy_data)

df = fetch_data(10)



with st.container():
	
	col1, col2, col3, col4 = st.columns([1, 0.25, 2, 2])
	selected_rows = []
	with col1:
		# HEADER COL 1 - PART 1
		st.subheader("1. Selecciona una ubicación")
		# TODO: Lista de ubicaciones para añadirselo al select. Cargado desde un JSON
		data = [{'location': 'London', 'code': 'uk'},{'location': 'Madrid', 'code': 'es'},{'location': 'USA', 'code': 'us'},{'location': 'Paris', 'code': 'pa'}]
		json_string = json.dumps(data)
		# Select box con elemento del json de localizaciones
		# TODO: Mirar como poner elemento of select box
		st.selectbox("Ubicaciones disponibles:", list(data), index=1)
		
		# HEADER COL 1 - PART 2
		st.subheader("2. Selecciona una tendencia")

		#df = pd.DataFrame([{"hashtag": ["#love"]}, {"hashtag": ["#love"]}, {"hashtag": ["#love"]}])
		options_builder = GridOptionsBuilder.from_dataframe(df)
		options_builder.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=True)
		options_builder.configure_column("hashtag", type=["stringColumn","stringColumnFilter"])
		options_builder.configure_selection("single", use_checkbox=True)
		options_builder.configure_pagination(paginationAutoPageSize=True)
		options_builder.configure_grid_options(domLayout='normal')
		grid_options = options_builder.build()
		grid_return = AgGrid(df, grid_options, update_mode="MODEL_CHANGED")

		selected_rows = grid_return["selected_rows"]
		
		# FORMULARIO HASHTAG
		#if len(selected_rows) == 0:
		#	hashtag = st.text_input('Introduce un hashtag', "#love")
		#else:
		#hashtag = st.text_input('Introduce un hashtag', selected_rows[0]["hashtag"])
		hashtag = selected_rows[0]["hashtag"]
				

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
query =  hashtag + " -is:retweet" 
response = client.search_recent_tweets(query=query,
                                     tweet_fields = ["created_at", "text", "lang", "source", "public_metrics", "entities"],
                                     user_fields = ["name", "username", "location", "verified", "description", "public_metrics"],
                                     max_results = 100,
                                     expansions='author_id'
                                     )

#Tweets formateo
# create a list of records
tweet_info_ls = []

# iterate over each tweet and corresponding user details
for tweet, user in zip(response.data, response.includes['users']):
    
    if "entities" in dict(tweet):
      # Extract info for hashtags
      if "hashtags" in dict(tweet.entities):
        hashtags = tweet.entities['hashtags']
        str_hashtags = ""
        if len(hashtags) > 1:
          for hashtag in hashtags:
            str_hashtags = str_hashtags + hashtag['tag'] + " | "
        else:
          for hashtag in hashtags:
            str_hashtags = hashtag['tag']
      else: str_hashtags = ""

      # Extract info for mentions
      if "mentions" in dict(tweet.entities):
        mentions = tweet.entities['mentions']
        str_mentions = ""
        if len(mentions) > 1:
          for mention in mentions:
            str_mentions = str_mentions + mention['username'] + " | "
        else:
          for mention in mentions:
            str_mentions = mention['username']
      else: str_mentions = ""
    else:
      str_hashtags = ""
      str_mentions = ""

    # Include tweet info in the csv
    tweet_info = {
        'created_at': tweet.created_at,
        'username': user.username,
        'location': user.location,
        'idioma': tweet.lang,
        'followers': user.public_metrics['followers_count'],
        'verified': user.verified,
        'text': tweet.text,
        'retweet_count': tweet.public_metrics['retweet_count'],
        'favorite_count': tweet.public_metrics['like_count'],
        'hashtags': str_hashtags,
        'user_mentions': str_mentions
    }
    tweet_info_ls.append(tweet_info)
    
    # Write in de csv
    #csvWriter.writerow([tweet.created_at, user.username, user.location, user.public_metrics['followers_count'], user.verified, tweet.text, tweet.public_metrics['retweet_count'], tweet.public_metrics['like_count'], str_hashtags, str_mentions])  

# create dataframe from the extracted records
tweets_df = pd.DataFrame(tweet_info_ls)


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


with col3:
	# HEADER COL 3
	st.header("3. Observa los resultados")
	
	#METRICA 1 - Tweets recogidos
	st.subheader("Tweets recogidos")
	tweets_recogidos = len(tweets_df)
	st.metric("Número de tweets recolectados", tweets_recogidos)
	
	
	#METRICA 2 - Alcance 
	st.subheader("Alcance")
	total_alcance = tweets_df['followers'].sum() + tweets_df['retweet_count'].sum()
	st.write("El alcance se calcula como el número de seguidores de las cuentas que han publicado el contenido recogido más el número de RTs de los tweets")
	st.metric("Usuarios potencialmente alcanzados", total_alcance)
	
	#METRICA 6 - Otros hashtags 
	st.subheader("Temas relacionados")
	st.write("Se indican los 20 hashtags más añadidos por los usuarios a los tweets del tema seleccionado")
	all_hashtags = []
	for i in tweets_df.index:
	  hashtags = tweets_df['hashtags'][i]
	  if hashtags != "":
	    hashtags = hashtags.split(" | ")
	    if len(hashtags) > 1:
	      hashtags.pop()
	      all_hashtags = all_hashtags + hashtags
	    else:
	      if hashtags[0] != "":
		all_hashtags = all_hashtags + hashtags
	fig6 = plt.figure()
	ax6 = fig6.add_axes([0,0,1,1])
	hashtags_df = pd.DataFrame(all_hashtags, columns =['hs'])
	labels6 = hashtags_df['hs'].drop_duplicates()
	labels6 = labels6[:20]
	labels6 = hashtags_df['hs'].reindex(index=labels.index[::-1])
	values6 = hashtags_df.value_counts()
	values6 = values6[:20]
	values6 = values6.sort_values(ascending=True)
	ax6.barh(labels6,values6)
	st.pyplot(fig6)
	

with col4:
	# HEADER COL 4
	st.header(" ")
	
	#METRICA 3 - Tweets en el tiempo 
	st.subheader("Número de tweets según hora de publicación")
	st.write("Se indica el número de tweets según la fecha en la que fueron publicados en Twitter")
	tweets_df['hora_minuto'] = tweets_df['created_at'].dt.strftime("%d/%m/%y %H:%M")
	x = tweets_df['hora_minuto'].drop_duplicates()
	y = tweets_df['hora_minuto'].value_counts()
	dif_minmax = datetime.strptime(max(x), '%d/%m/%y %H:%M') - datetime.strptime(min(x), '%d/%m/%y %H:%M')
	rango = int(dif_minmax.total_seconds() / 60)
	x_ticks = [0,rango]
	x_labels = [min(x),max(x)]
	fig3 = plt.figure()
	plt.plot(x, y)
	plt.xticks(ticks=x_ticks, labels=x_labels)
	st.pyplot(fig3)
	
	#METRICA 4 - Tweets según idioma
	st.subheader("Número de tweets según idioma de publicación")
	st.write("Se indica el idioma detectado en el texto del tweet")
	fig4 = plt.figure()
	ax4 = fig4.add_axes([0,0,1,1])
	labels4 = tweets_df['idioma'].drop_duplicates()
	labels4 = labels4.reindex(index=labels4.index[::-1])
	values4 = tweets_df['idioma'].value_counts().sort_values(ascending=True)
	ax4.barh(labels4,values4)
	st.pyplot(fig4)
	
	#METRICA 5 - Sentimiento
	st.subheader("Distribución del sentimiento")
	st.write("Distribución del sentimiento detectado en los tweets")
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

	
	
	
	
	
	
	
