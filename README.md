filmes.pynb
https://colab.research.google.com/drive/1IF2LTpCn2lyq9eeWCeLjz7ViNS00XOF8?authuser=4#scrollTo=gIwmS_Z3_us_
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

drive.mount('/content/gdrive')

filmes = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/Dados/9k_movies.csv')

filmes.isnull().sum()

filmes.isna().sum()

filmes

filmes.columns

filmes.dtypes

filmes.shape

# Tratamento

# Removendo o 'm' dos dados da coluna "Duration" para ficar mais fácil de filtrar

filmes['Duration'] = filmes['Duration'].str.replace('m', '', regex=False)
filmes['Duration'] = filmes['Duration'].fillna(0).astype(int)

Converter o valor de "genres" para apenas o primeiro gênero do filme (para lidarmos com string ao invés de array

def first_genre(genre_str):
    try:
        genre_list = genre_str.strip('[]').split(',')
        return genre_list[0].strip().strip("'") if genre_list else None
    except AttributeError:  # Excessão
        return None

filmes['Genre'] = filmes['Genre'].apply(first_genre)

Removendo os valores nulos da nota do IMDB

filmes = filmes.dropna(subset=['IMDB_rating'])

filmes.isnull().sum()

filmes.columns

filmes

---
Notei que alguns gêneros estão vazios

---

# Encontrar valores de gênero que estejam vazios
emptyGenres = filmes[filmes['Genre'] == ""]

numEmptyGenres = len(emptyGenres)
print(f"Número de gêneros vazios: {numEmptyGenres}")

filmes = filmes[filmes['Genre'] != ""]

filmes

# Queries

---
Qual a Probabilidade de ser um filme feito nos Estados Unidos?

---

probUS = len(filmes[filmes['Country']== 'us']) / len(filmes)

print('A probabilidade de ser um filme produzido nos Estados Unidos é de: ' + str(round(probUS*100,4)) + '%')

---
Qual a probabilidade de ser um filme com no minimo de 80 de duração **ou** lançado em 2020

---

tamtemFil = len(filmes[(filmes['Duration'] >= 80) | (filmes['Time'] == 2020)])

probTamTem = round(tamtemFil/len(filmes), 4)

print('A probabilidade de ser um filme com no mínimo 80 minutos de duração ou ter sido lançado em 2020 é de: ' + str(round(probTamTem*100,4)) + '%')

---
Qual a probabilidade de ser um filme com no minimo de 80 de duração **e** ser lançado em 2020

---

temtamAnd = len(filmes[(filmes['Duration'] >= 80) & (filmes['Time'] == 2020)])

probTemTam = round(temtamAnd/len(filmes), 4)

print('A probabilidade de ser um filme com no mínimo 80 minutos de duração e ter sido lançado em 2020 é de: ' + str(round(probTemTam*100,4)) + '%')

---
Qual a probabilidade de um filme de Drama ter uma nota maior que 7

---

generoNota7Drama = len(filmes[(filmes['Genre'] == 'drama') & (filmes['IMDB_rating'] >= 7)])
probDrama7 = round(generoNota7Drama/len(filmes), 4)
print('A probabilidade de ser um filme de Drama com uma nota maior que 6 é de: ' + str(round(probDrama7*100,4)) + '%')

# Probabilidade Condicional

tempFil = len(filmes[filmes['Duration']>= 80])

probTemp = round(tempFil/len(filmes), 4)

print('A probabilidade de ser um filme com mais de 80 minutos é de: ' + str(round(probTemp*100,4)) + '%')

lancFil = len(filmes[filmes['Time'] == 2020])

probLanc = round(lancFil/len(filmes), 4)

print('A probabilidade de ser um filme lançado em 2020 é de: ' + str(round(probLanc*100,4)) + '%')

#Probabilidade de ser um filme de 80 min+ e ter sido lançado em 2020 sobre a probabilidade de um filme ter 80 min+
probCond = probTemTam/probTemp

print('A probabilidade de ser um filme lançado em 2020 sabendo que tem no mínimo 80 min de duração: ' + str(round(probCond*100,4)) + '%')

# Probabilidade condicional de ser um filme de drama dado que a nota é >= 7

# Número de filmes com nota >= 7
numNota7 = len(filmes[filmes['IMDB_rating'] >= 7])

probDramaNota7 = generoNota7Drama / numNota7

print("A probabilidade condicional de ser um filme de drama, dado que a nota é no mínimo 7, é: " + str(round(probDramaNota7*100,2)) + '%')

"Se o filme tem uma nota maior que 7, pode-se considerar que o público achou ele bom, como a chance de ser um filme de drama é decente considerando a quantidade de gêneros que tem no dataset, dá para se dizer que o público gosta de drama"

---
#Teorema de Bayes

---

probTemTotal = (probCond*probTemp)/probLanc
print('A probabilidade de ser um filme com mais de 80 minutos, tal que tenha sido lançado em 2020 é de: ' + str(round(probTemTotal*100,4)) + '%')

probDrama = round(len(filmes[filmes['Genre'] == 'drama']) / len(filmes),4)
probNota7 = round(len(filmes[filmes['IMDB_rating'] >= 7]) / len(filmes),4)

probNotaTotal = (probDramaNota7*probDrama)/probNota7

print("A probabilidade de ser um filme de drama, tal que a nota seja no mínimo 7, é: " + str(round(probNotaTotal*100,2)) + '%')

---
#Variáveis Aleatórias
---

filmes['Genre'].value_counts()

filmes['Genre'].value_counts().sort_index()

#Variáveis Aleatórias de filmes

ImdbArredondado = filmes['IMDB_rating'].round(0).astype(int)

ft = ImdbArredondado.value_counts(sort=False).sort_index()/filmes['IMDB_rating'].count()
print(ft)

fp = filmes['Genre'].value_counts(sort=False).sort_index()/filmes['Genre'].count()
print(fp)

# Função Repartição (cumulativa)

def prob_cumulativa(genre1, genre2):
  prob_genre1 = len(filmes[filmes['Genre'] == genre1]) / len(filmes)
  prob_genre2 = len(filmes[filmes['Genre'] == genre2]) / len(filmes)
  cumulative_prob = prob_genre1 + prob_genre2
  return cumulative_prob


comedy_romance_prob = prob_cumulativa("comedy", "romance")
print("A probabilidade cumulativa de um filme ser o gênero Comédia ou Romance é: " + str(round(comedy_romance_prob, 4)))

A chance de eu selecionar um filme aleatoriamente e ele ser de comédia ou de romance (o valor máximo é 1)

#Gráficos

plt.figure(figsize=(12, 6))
sns.countplot(x='Genre', data=filmes)
plt.xticks(rotation=50, ha='right')
plt.xlabel('Gênero do Filme')
plt.ylabel('Número de Filmes')
plt.title('Distribuição de Gêneros de Filmes')
plt.show()

#Box Plot

from matplotlib import pyplot as plt
import plotly.express as px

fig=px.box(filmes,x='Genre',y='IMDB_rating',color='Genre')
fig.show()

Visualização detalhada de média, mediana, moda, outliers, mínimos e máximos

documentarios = filmes[filmes['Genre'] == 'documentary']
MaiorRatingDoc = documentarios.loc[documentarios['IMDB_rating'].idxmax()]

MaiorRatingDoc

# HeatMap

# Criando uma tabela pivô para o heatmap, para definir as colunas utilizadas, a função e organizar os valores de forma crescente
heatmap_data = filmes.pivot_table(index='Genre', values=['IMDB_rating'], aggfunc='mean').sort_values(by='IMDB_rating', ascending=False)
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Avaliação Média IMDB por Gênero')
plt.show()

# Scatter plot

plt.figure(figsize=(10, 6))
plt.scatter(filmes['Duration'], filmes['IMDB_rating'])
plt.xlabel('Duração')
plt.xticks(rotation=50, ha='right')
plt.ylabel('Nota IMDB')
plt.title('Nota IMDB vs. Duração')
plt.grid(True)
plt.show()

# Histogramas

plt.figure(figsize=(10, 6))
plt.hist(filmes['IMDB_rating'], bins=20, edgecolor='black')
plt.xlabel('Notas no IMDB')
plt.ylabel('Frequencia')
plt.title('Distribuição das notas no IMDB')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(filmes['Genre'], bins=20, edgecolor='black')
plt.xlabel('Gênero')
plt.xticks(rotation=50, ha='right')
plt.ylabel('Frequência')
plt.title('Distribuição de Gêneros')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
filmes_acima_7 = filmes[filmes['IMDB_rating'] > 7]
plt.hist(filmes_acima_7['Time'], bins=20, edgecolor='black')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Frequência de Filmes (Nota > 7)')
plt.title('Frequência de Filmes com Nota Maior que 7 ao Longo dos Anos')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
filmes_acao = filmes[filmes['Genre'] == 'action']
plt.hist(filmes_acao['Time'], bins=20, edgecolor='black')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Frequência de Filmes de ação')
plt.title('Frequência de Filmes de ação ao Longo dos Anos')
plt.grid(True)
plt.show()

Essa Visualização foi feita para ajudar a entender as notas mais comuns junto dos gêneros mais comuns no dataset, para que possa ser vista se tem uma relação ou não. E por fim analisar como anda o desempenho dos filmes ao longo dos anos

#Distribuições Amostrais

sample_size1 = 10
sample1 = filmes.sample(n=sample_size1, random_state = 42)

# Informações da Amostra
print(sample1.describe())
print(sample1['IMDB_rating'].mean())
print(sample1['IMDB_rating'].median())

plt.figure(figsize=(10, 6))
plt.hist(sample1['IMDB_rating'], bins=10, edgecolor='black')
plt.xlabel('Avaliação IMDB')
plt.ylabel('Frequência')
plt.title('Histograma da Avaliação IMDB da Amostra (10 Amostras)')
plt.show()

# Comparando a média da amostra com a média do dataset
Media_notas1 = filmes['IMDB_rating'].mean()
Mediana_notas1 = filmes['IMDB_rating'].median()
Media_notasamostra1 = sample1['IMDB_rating'].mean()
Mediana_notasamostra1 = sample1['IMDB_rating'].median()

print(f"Média das notas: {Media_notas1:,.4f}")
print(f"Mediana das notas: {Mediana_notas1:,.4f}")
print(f"Média da amostra: {Media_notasamostra1:,.4f}")
print(f"Mediana da amostra: {Mediana_notasamostra1:,.4f}")

sample_size2 = 100
sample2 = filmes.sample(n=sample_size2, random_state = 42)


print(sample2.describe())
print(sample2['IMDB_rating'].mean())
print(sample2['IMDB_rating'].median())

plt.figure(figsize=(10, 6))
plt.hist(sample2['IMDB_rating'], bins=10, edgecolor='black')
plt.xlabel('Avaliação IMDB')
plt.ylabel('Frequência')
plt.title('Histograma da Avaliação IMDB da Amostra (100 Amostras)')
plt.show()

Media_notas2 = filmes['IMDB_rating'].mean()
Mediana_notas2 = filmes['IMDB_rating'].median()
Media_notasamostra2 = sample2['IMDB_rating'].mean()
Mediana_notasamostra2 = sample2['IMDB_rating'].median()

print(f"Média das notas: {Media_notas2:,.4f}")
print(f"Mediana das notas: {Mediana_notas2:,.4f}")
print(f"Média da amostra: {Media_notasamostra2:,.4f}")
print(f"Mediana da amostra: {Mediana_notasamostra2:,.4f}")

sample_size3 = 500
sample3 = filmes.sample(n=sample_size3, random_state = 42)

print(sample3.describe())
print(sample3['IMDB_rating'].mean())
print(sample3['IMDB_rating'].median())

plt.figure(figsize=(10, 6))
plt.hist(sample3['IMDB_rating'], bins=10, edgecolor='black')
plt.xlabel('Avaliação IMDB')
plt.ylabel('Frequência')
plt.title('Histograma da Avaliação IMDB da Amostra (500 Amostras)')
plt.show()

Media_notas3 = filmes['IMDB_rating'].mean()
Mediana_notas3 = filmes['IMDB_rating'].median()
Media_notasamostra3 = sample3['IMDB_rating'].mean()
Mediana_notasamostra3 = sample3['IMDB_rating'].median()

print(f"Média das notas: {Media_notas3:,.4f}")
print(f"Mediana das notas: {Mediana_notas3:,.4f}")
print(f"Média da amostra: {Media_notasamostra3:,.4f}")
print(f"Mediana da amostra: {Mediana_notasamostra3:,.4f}")

# Extra: Buscando a Duração média de duração dos Filmes durantes os anos

filmes['Time'].value_counts().sort_index()

plt.figure(figsize=(12, 8))
sns.histplot(data=filmes, x='Time', y='Duration', bins=30)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Duração Média (minutos)')
plt.title('Histograma da Duração Média dos Filmes ao Longo dos Anos')
plt.show()

#Outlier explícito em 2021, além do gráfico lembrar bastante uma Distribuição normal

dur2021 = filmes[filmes['Time'] == 2021]
MaiorDuration = dur2021.loc[dur2021['Duration'].idxmax()]
MaiorDuration

Buscando por esse filme, descobri que se trata apenas de um erro de digitação, o filme originalmente tem 81 minutos (1 hora e 21 minutos) de duração
