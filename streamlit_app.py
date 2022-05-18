

import streamlit as st
import pandas as pd
#import plotly.figure_factory as ff
import tweepy
import matplotlib.pyplot as plt
import numpy as np


st.title('Twitter prueba')

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.area_chart(chart_data)

# Gráfico tarta

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

labels = 'Positive', 'Neutral', 'Negative'
sections = [50, 35, 15]
colors = ['g', 'y', 'r']

#fig = plt.pie(sections, labels=labels, colors=colors,
#        startangle=90,
#        explode = (0, 0, 0),
#        autopct = '%1.2f%%')

#plt.axis('equal') # Try commenting this out.
#plt.title('Pie Chart Twitter Sentiment Example')
#plt.show()

st.pyplot(fig)
