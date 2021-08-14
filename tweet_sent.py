# Instalación de la biblioteca twint
'''
Librería avanzada de python que se utiliza para extraer tweets
en forma de dataframe. Además, no requiere credencial de autenticación para
conectarse a Twitter.
'''

!pip install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
!pip install textblob
!pip install wordcloud
# Librerías

## Extracción de tweets
import twint
## Funciones algebráicas
import numpy as np
## Tratamiento de datos
import pandas as pd
## Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
## Para crear la columna de polaridad
from textblob import TextBlob
## Nube de palabras
from wordcloud import WordCloud,ImageColorGenerator
## llamar rutinas asíncronas, similar a realizar tareas concurrentes.
import nest_asyncio
nest_asyncio.apply()
## Expresiones regulares
import re
## Manejo de fechas
import calendar
## Eliminación de los avisos
import warnings
warnings.filterwarnings('ignore')


# Declaración del objeto de twint
'''
Se buscan tweets que se refieran a la vinculación de mbappe y real madrid

'''

c=twint.Config()
c.Search='mbappe real madrid'
c.Since='2021-07-11'
c.Until='2021-08-11'
c.Hide_output=True
c.Pandas=True
twint.run.Search(c)
## Se guardan resultados en un dataframe
df=twint.storage.panda.Tweets_df
df
df.info()


# Extracción del año mes y día de la columna fecha
## Año
df['year']=pd.to_datetime(df['date']).dt.strftime('%Y')
## Mes
df['month']=pd.to_datetime(df['date']).dt.strftime('%m')
## Día
df['dayweek']=pd.to_datetime(df['date']).dt.strftime('%A')



df

# Procesamiento de los Tweets

'''
Una vez que tengamos nuestras columnas listas, procesemos previamente los tweets
(es decir, eliminando las URL, el nombre de usuario y las palabras vacías), ya
que no agregan valor al sentimiento. Para esto crearemos una función y la llamaremos
usando el método lambda. Aquí, estoy usando un archivo stopwords.txt que contiene
la lista de palabras vacías que se deben eliminar.
'''

## Declaración de la función de procesado de tweets

def preprocess_tweets(tweet):
    ## Se define la fuente de stopwords
    ## "r+"" abre un archivo en modo lectura y escritura
    fo = open('stopwords.txt', 'r+')
    ## se declara la variable stopwords como lista
    stop_words = list(fo.read().split(','))
    ## Creación de tabla de traducción manual utilizando ASCII
    translation={39:None}
    ## Declaración del tweet procesado utilizando expresiones regulares (re)
    ## se utiliza el método sub(substitución) y definen caracteres y rangos de caracteres a sustituir
    processed_tweet=' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)',' ',tweet).split())
    ## Se seleccionan las palabras que no sean stopwords
    processed_tweet = ' '.join(word for word in processed_tweet.split() if word not in str(stop_words).translate(translation))
    return processed_tweet


df['processed_tweet']=df['tweet'].apply(lambda x:preprocess_tweets(x.lower()))

df['processed_tweet']


# Clasificación de tweets en función del sentimiento.

## Creación de la columna polaridad (valores numéricos)

df['polarity']=df[df['language']=='en']['processed_tweet'].apply(lambda x:TextBlob(x).sentiment[0])

## Creación de la columna sentimiento (positivo, negativo, neutro)

### Creación de un nuevo set de datos con solo los tweets en inglés

data=df[df['language']=='en']

data

data['sentiment']=data['polarity'].apply(lambda x: 'positive' if x>0 else('negative' if x<0 else 'neutral'))

data[['created_at', 'date', 'username', 'tweet', 'language', 'sentiment']]


# Comprobación de como están distribuidos los tweets en función del sentimiento

sentiment_count=data['sentiment'].value_counts(normalize=True)*100

sentiment_count.reset_index()
sentiment_count=pd.DataFrame(sentiment_count)
sentiment_count=sentiment_count.reset_index()
sentiment_count=sentiment_count.rename(columns={'sentiment':'percentage[%]', 'index':'sentiment'})
sentiment_count


plt.figure(figsize=(10,7))
plt.title('Tweets classification based on sentiment', fontsize=18, weight='bold')
plt.xlabel('Sentiment', fontsize=14, weight='bold')
plt.ylabel('Percentage [%]', fontsize=14, weight='bold')
plt.grid(color='#95a5a6', linestyle='-.', linewidth=1, axis='y', alpha=0.7)
ax=sns.barplot(data=sentiment_count, x='sentiment', y='percentage[%]', palette='Reds_r')
ax.set_yticks(np.arange(0,110,10))
ax2=ax.twinx()
ax2.set_yticks(np.arange(0,110,10)*len(data)/100)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()), (p.get_x()+0.30, p.get_height()+1), weight='bold')
plt.show()

'''
La clasificación de tweets en base a sentimiento muestra que los mensajes tienen
mayoritariamente un tono neutral (68,71%).
'''

# Observación como se concentran los tweets a lo largo del periodo establecido
## Creación de un dataframe donde las fechas y el sentimiento estén agrupados
df_sent=data.groupby(by=['date', 'sentiment']).agg({'tweet':'count'}).reset_index()
df_sent
'''
Como se puede ver, la fecha también presente la hora en la que se ha generado el tweet.
En este caso, el interés está en simplificar el dataframe por fecha en formato
año, mes y día. Por ello, se va a formatear la fecha y se va a volver a agrupar
'''
df_sent['date']=pd.to_datetime(df_sent['date']).dt.strftime('%Y-%m-%d')
df_sent=df_sent.groupby(by=['date', 'sentiment']).agg({'tweet':'count'}).reset_index()
df_sent
## Concentración de tweets por fecha en base al sentimiento
plt.figure(figsize=(15,7))
sns.lineplot(data=df_sent, x='date', y='tweet', hue='sentiment', palette=['red', 'blue', 'green'])
plt.xticks(rotation=90)
plt.title('Tweets by date based on sentiment', weight='bold', fontsize=18)
plt.xlabel('Date', weight='bold', fontsize=14)
plt.ylabel('Number of tweets', weight='bold', fontsize=14)
plt.show()

## Concentración de tweets por fecha
### Creación de un nuevo dataframe que no contenga la variable sentimiento

df_tweet=df_sent[['date', 'tweet']]

df_tweet=df_tweet.groupby(by=['date'])['tweet'].sum().reset_index()

df_tweet
plt.figure(figsize=(15,7))
sns.lineplot(data=df_tweet, x='date', y='tweet', color='crimson')
plt.xticks(rotation=90)
plt.title('Tweets by date ', weight='bold', fontsize=18)
plt.xlabel('Date', weight='bold', fontsize=14)
plt.ylabel('Number of tweets', weight='bold', fontsize=14)
plt.show()

'''
Se puede observar claramente que la mayor concentración de tweets se produjo el día 5, 6 de agosto,
coincidiendo con la confirmación de que Messi dejaba el Barcelona, y el día 10 de agosto, cuando se produjo la confirmación
de su fichaje por el PSG. Sin duda, estos tweets muy posiblemente estarán relacionados con especulaciones sobre su futuro.
'''

# Creación de nube de palabras: comprobación de que palabras son las más utilizadas durante esos días.

## Creación de un nuevo set de datos
data.columns
df_wc=data[['date', 'processed_tweet']]

df_wc
df_wc['date']=pd.to_datetime(df_wc['date']).dt.strftime('%Y-%m-%d')

df_wc


## Filtrado de fechas
df_wc_date=df_wc[(df_wc['date']=='2021-08-05')|(df_wc['date']=='2021-08-06')|(df_wc['date']=='2021-08-10')]
df_wc_date


## Creación de la nube de palabras de los días 5, 6 y 10 de agosto

### Declaración del texto que se va a utilizar en la nube de palabras
text=' '.join(tweet for tweet in df_wc_date['processed_tweet'].astype(str))
### Declaración de la nube de palabras
wordcloud=WordCloud(background_color='white',
                    width=1000,
                    height=500,
                    max_words=25).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.rcParams['figure.figsize']=[20,20]
plt.tight_layout()
'''
Como se puede observar en la nube de palabras, se destaca la asociación
de mbappe con el real madrid, lo que indica que durante esos días se ha
estado especulando de forma muy intensa con su futuro.

Habrá que estar pendiente de lo que puede ocurrir con su futuro
en las próximas semanas.
'''
