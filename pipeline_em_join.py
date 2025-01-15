# Funções para etapa de pré-processamento
# Funções para etapa de representação dos dados / treinamento
# Funções para etapa de junção
# PIPELINE EXEC

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

### Importar bibliotecas
import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time, sys, os, subprocess
import ipywidgets as widgets
import math
import random
import re
import torch
import faiss
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation

## Biblitecas para pré-processamento do texto
import nltk
from nltk.stem.snowball import SnowballStemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
import nltk
nltk.download('punkt')
import spacy
nlp_spacy = spacy.load("en_core_web_sm")
# cria um objeto EntityRecognizer a partir do modelo
#entity_recognizer = nlp_spacy.get_pipe('ner')

from torch.utils.data import DataLoader

import inspect
import logging
from sentence_transformers import LoggingHandler
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    force=True,
                    handlers=[LoggingHandler()])

from datasets import Dataset

# Funções para etapa de pré-processamento

def get_var_name(var):
  """
  Retorna o nome da variável como string
  """
  for fi in reversed(inspect.stack()):
    names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
    if len(names) > 0:
      return names[0]

def carregar_dados(url):
  df = pd.read_csv(url,sep='\t',header = None)
  df[2].fillna("", inplace = True)
  df[2] = df[2].str.replace('\d+', '')
  df[3] = df[1] + '[SEP]' + df[2]
  return df

def remove_numbers_in_text(text):
  return re.sub(r'\d+', '', text)

def remove_special_chars_in_text(text):
  return re.sub(r'[^\w\s]', '', text)

# instancie o stemmer
stemmer = SnowballStemmer("english")

# stop words
stop_words = get_stop_words('english')

# defina a função de stemização
def stem_text(texto):
  # tokeniza o texto
  tokens = nltk.word_tokenize(texto, language="english")
  # aplica a stemização em cada token
  stemmed_tokens = [stemmer.stem(token) for token in tokens]
  # junta os tokens stemizados em um novo texto
  texto_stemizado = " ".join(stemmed_tokens)
  # retorna o texto stemizado
  return texto_stemizado

# defina função de lematização
def lem_text(text):
  doc = nlp_spacy(text)
  lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
  return ' '.join(lemmas)

# defina a função de sumarização
def sum_text(texto,percent):
  # obtém as stop words para o idioma inglês
  stop_words = get_stop_words("english")
  # realiza a sumarização do texto utilizando a biblioteca sumy
  parser = PlaintextParser.from_string(texto, Tokenizer("english"))

  # definir tamanho máximo
  sentence_count = len(parser.document.sentences)
  summary_size = math.ceil(sentence_count * percent)

  #summarizer = LsaSummarizer()
  #summarizer = TextRankSummarizer()
  summarizer = LexRankSummarizer()

  summarizer.stop_words = stop_words # define as stop words a serem removidas
  summary = summarizer(parser.document, summary_size)
  resumo = ""
  for sentence in summary:
    resumo += str(sentence) + " "
  # retorna o resumo do texto
  return resumo


def aum_text(texto):
  doc = nlp_spacy(texto)
  novo_texto = ""
  for token in doc:
    if token.ent_type_:
      tipo_entidade = token.ent_type_
      novo_texto += f"{tipo_entidade} {token.text} "
    else:
      novo_texto += f"{token.text} "
  return novo_texto.strip()


# unir os atributos em uma só coluna
def transferir_valores(df, separador,remove_numbers,remove_special_chars,stem_sentence,lemmatize_sentence, nome_nova_coluna='sentence'):
  var_name = get_var_name(df)
  if remove_numbers:
    print("-> Removendo números de", var_name)
    df = df.astype(str).applymap(remove_numbers_in_text)
  if remove_special_chars:
    print("-> Removendo caracteres especiais de", var_name)
    df = df.astype(str).applymap(remove_special_chars_in_text)

  if stem_sentence:
    print("-> Processando stemização em",var_name)
    df = df.astype(str).applymap(stem_text)

  if lemmatize_sentence:
    print("-> Processando lemmatização em",var_name)
    df = df.astype(str).applymap(lem_text)

  if data_augmentation:
    print("-> Realizando aumento de dados com reconhecimento de entidades em",var_name)
    df = df.astype(str).applymap(aum_text)

  print("-> Agrupando atributos em única sentença para",var_name)
  nova_coluna = df.apply(lambda x: separador.join(x.astype(str)), axis=1)

  return pd.DataFrame(nova_coluna, columns=[nome_nova_coluna])

## Carregar tableA, tableB, train, valid e test em dataframes
def process_tables_labels(csv_tableA, csv_tableB, csv_train, csv_valid, csv_test, token_separator, NaN_substitute,remove_numbers,remove_special_chars,stem_sentence,lemmatize_sentence,summarize_sentence,percent_summarize,data_augmentation):
  df_tableA = pd.read_csv(csv_tableA,sep=',',header = 0)
  df_tableB = pd.read_csv(csv_tableB,sep=',',header = 0)

  #armazenar o id em outro dataframe
  idsA = df_tableA['id']
  idsB = df_tableB['id']
  dfIdsA = idsA.to_frame()
  dfIdsB = idsB.to_frame()

  #remover coluna id
  df_tableA = df_tableA.drop(columns=df_tableA.columns[0])
  df_tableB = df_tableB.drop(columns=df_tableB.columns[0])

  #setar valor para atributos sem conteúdo
  df_tableA.fillna(NaN_substitute, inplace = True)
  df_tableB.fillna(NaN_substitute, inplace = True)

  dfSentenceA = pd.DataFrame()
  dfSentenceB = pd.DataFrame()
  dfSentenceA['sentence'] = transferir_valores(df_tableA,token_separator,remove_numbers,remove_special_chars,stem_sentence,lemmatize_sentence,data_augmentation)
  dfSentenceB['sentence'] = transferir_valores(df_tableB,token_separator,remove_numbers,remove_special_chars,stem_sentence,lemmatize_sentence,data_augmentation)

  if summarize_sentence:
    print("-> Processando summarização das sentenças")
    dfSentenceA['sentence'] = dfSentenceA['sentence'].apply(lambda x: sum_text(x, percent_summarize))
    dfSentenceB['sentence'] = dfSentenceB['sentence'].apply(lambda x: sum_text(x, percent_summarize))
    #dfSentenceA['sentence'] = dfSentenceA['sentence'].apply(sum_text)
    #dfSentenceB['sentence'] = dfSentenceB['sentence'].apply(sum_text)

  #train
  dfTrain = pd.read_csv(csv_train,sep=',',header = 0)
  #valid / eval
  dfValid = pd.read_csv(csv_valid,sep=',',header = 0)
  #test
  dfTest = pd.read_csv(csv_test,sep=',',header = 0)

  return (dfIdsA, dfIdsB, dfSentenceA, dfSentenceB, dfTrain, dfValid, dfTest)

## Carregar tabela única em dataframe
def process_table(tsv_table, token_separator, NaN_substitute,remove_numbers,remove_special_chars,stem_sentence,lemmatize_sentence,summarize_sentence,percent_summarize,data_augmentation):
  df_table = pd.read_csv(tsv_table,sep='\t',header = 0)

  #armazenar o id em outro dataframe
  ids = df_table['id']
  dfIds = ids.to_frame()

  #remover coluna id
  df_table = df_table.drop(columns=df_table.columns[0])

  #setar valor para atributos sem conteúdo
  df_table.fillna(NaN_substitute, inplace = True)

  dfSentence = pd.DataFrame()
  dfSentence['sentence'] = transferir_valores(df_table,token_separator,remove_numbers,remove_special_chars,stem_sentence,lemmatize_sentence,data_augmentation)


  if summarize_sentence:
    print("-> Processando summarização das sentenças")
    dfSentence['sentence'] = dfSentence['sentence'].apply(lambda x: sum_text(x, percent_summarize))

  return (dfIds, dfSentence)

# Funções para etapa de representação dos dados / treinamento

def gera_input_examples(dfRotulos,dfA,dfB,dfIdsA,dfIdsB):
  examples = []
  for d in dfRotulos.index:
    idLeft = dfRotulos['ltable_id'][d]
    idRight = dfRotulos['rtable_id'][d]

    #Substituir o valor do ID pela posição, caso id não seja a posição (ex: dataset company)
    if dfIdsA.iloc[0]['id'] != 0 and dfIdsA.iloc[(len(dfIdsA)-1)]['id'] != len(dfIdsA)-1:
      idLeft = dfIdsA[dfIdsA['id'] == idLeft].index[0]
      idRight = dfIdsB[dfIdsB['id'] == idRight].index[0]

    currentSentenceA = dfA.sentence[idLeft]
    currentSentenceB = dfB.sentence[idRight]
    currentLabel = int(dfRotulos['label'][d])

    examples.append(InputExample(texts=[currentSentenceA, currentSentenceB], label=currentLabel))
  return examples

def gera_vetores(dfA,dfB,model,normalize_embedings):
  vetoresA = model.encode(dfA,show_progress_bar=True,normalize_embeddings=normalize_embedings)
  vetoresB = model.encode(dfB,show_progress_bar=True,normalize_embeddings=normalize_embedings)

  return vetoresA,vetoresB

# Funções

# 1 - Realizar junção para encontrar a similaridade entre os pares
# A entrada são (dfTable,dfIds) onde dfTable são os registros com a coluna ('sentence') e dfIds o id de cada registro, os registros com o mesmo id significam que são duplicatas.
# usar faiss hnsw (parâmetros que melhorem a performance)
# usar k (k=20)
# com o resultados pronto no seguinte formato (pos_l, pos_r, ltable_id, rtable_id, score) classificar em ordem decrescente.
# Onde pos_l e pos_r são a posição do vetor na lista de vetores e ltable_id e rtable_id são o id do registro contido no dfIds

# gerar vetores
def gera_vetores_one_table(model,dfTable,normalize_embedings):
  embeddings = model.encode(dfTable['sentence'],show_progress_bar=True,normalize_embeddings=normalize_embedings)
  return embeddings

# fazer junção com faiss
def find_similar_pairs(embeddings, dfIds, k):
  """
  Realiza a junção dos registros para encontrar similaridade entre pares e retorna
  os resultados ordenados em ordem decrescente.
  """

  # Cria índice HNSW com os vetores
  d = embeddings.shape[1]
  M = 32
  efC = 16
  efS = 16
  index = faiss.IndexHNSWFlat(d,M,faiss.METRIC_INNER_PRODUCT)
  index.hnsw.efConstruction = efC
  index.hnsw.efSearch = efS
  index.add(embeddings)

  resultados = []
  i = 0
  for vetor in vetores:
    vetor_consulta = vetor.reshape(1,-1).astype(np.float32)
    D, I = index.search(vetor_consulta,k = k)
    for result,prod in zip(I.reshape(-1,1),D.reshape(-1,1)):
      if result > i: # não incluir o vetor corrente no resultado
        resultados.append( [i,result[0],dfIds.id[i],dfIds.id[result[0]],prod[0]])
    i = i + 1

  dfResults = pd.DataFrame(resultados, columns=['pos_l', 'pos_r', 'ltable_id', 'rtable_id', 'score'])

  # Ordena os resultados em ordem decrescente de score
  dfResults = dfResults.sort_values(by=['score'], ascending=False)

  return dfResults

# 2 - Ggerar positivos difíceis e negativos difíceis: pegar os pares que possuem baixa similaridade (abaixo de 0.6) porém são duplicatas.
# Para isso, percorrer cada resultado abaixo de 0.6 e verificar se o ltable_id == rtable_id. Se sim, incluir na lista dos positivos difíceis (duplicatas).
# A relação deverá ser no formato (ltable_id, rtable_id, label) onde label será 0 (não duplicata) e 1 (duplicata)
def generate_hard_labels(dfResults, pos_threshold=0.6, neg_threshold=0.8):
  """
  Gera pares positivos difíceis a partir dos resultados da junção, considerando
  os pares que possuem baixa similaridade (abaixo do threshold) porém são duplicatas.
  """
  hard_positives = []
  hard_negatives = []

  for index, row in dfResults.iterrows():
    pos_l = int(row['pos_l'])
    pos_r = int(row['pos_r'])
    ltable_id = row['ltable_id']
    rtable_id = row['rtable_id']
    score = row['score']

    if score < pos_threshold and ltable_id == rtable_id:
      hard_positives.append((pos_l, pos_r, 1))

    if score > neg_threshold and ltable_id != rtable_id:
      hard_negatives.append((pos_l, pos_r, 0))

  hard_positives_df = pd.DataFrame(hard_positives, columns=['ltable_id', 'rtable_id', 'label'])
  hard_negatives_df = pd.DataFrame(hard_negatives, columns=['ltable_id', 'rtable_id', 'label'])

  return hard_positives_df,hard_negatives_df

## Gerar pares difíceis juntamente com pares fáceis para reduzir viés do treinamento.
# Já faz o balanceamento na mesma quantidade de pares difíceis e fáceis, positivos e negativos.
def generate_balanced_labels(dfResults, pos_threshold=0.6, neg_threshold=0.8):
  """
  Gera pares positivos e negativos difíceis e fáceis a partir dos resultados da junção.
  """
  hard_positives = []
  hard_negatives = []
  easy_positives = []
  easy_negatives = []

  for index, row in dfResults.iterrows():
    pos_l = int(row['pos_l'])
    pos_r = int(row['pos_r'])
    ltable_id = row['ltable_id']
    rtable_id = row['rtable_id']
    score = row['score']

    if score < pos_threshold and ltable_id == rtable_id:
      hard_positives.append((pos_l, pos_r, 1))

    elif score > neg_threshold and ltable_id != rtable_id:
      hard_negatives.append((pos_l, pos_r, 0))

    elif score > neg_threshold and ltable_id == rtable_id:
      easy_positives.append((pos_l, pos_r, 1))

    elif score < pos_threshold and ltable_id != rtable_id:
      easy_negatives.append((pos_l, pos_r, 0))

  # Seleciona a mesma quantidade de pares fáceis e difíceis, ajustando para o que tiver menor quantidade
  min_len_pos = min(len(hard_positives), len(easy_positives))
  min_len_neg = min(len(hard_negatives), len(easy_negatives))
  min_len = min(min_len_pos, min_len_neg)

  # Balanceia a quantidade entre positivos e negativos, utilizando under-sampling
  hard_positives = random.sample(hard_positives, min_len)
  hard_negatives = random.sample(hard_negatives, min_len)
  easy_positives = random.sample(easy_positives, min_len)
  easy_negatives = random.sample(easy_negatives, min_len)

  positives_labels = pd.DataFrame(easy_positives + hard_positives, columns=['ltable_id', 'rtable_id', 'label'])
  negative_labels = pd.DataFrame(easy_negatives + hard_negatives, columns=['ltable_id', 'rtable_id', 'label'])

  return positives_labels, negative_labels

# 3 - Verificar o tamanho das listas (positivos difíceis e negativos difíceis) e fazer o balanceamento para que fiquem com tamanhos semelhantes.
# Ou seja, reduzir a que estiver maior para que fique no máximo 5% maior que a outra.
def balance_pos_neg(hard_positives, hard_negatives):
  """
  Balanceia as listas de positivos difíceis e negativos difíceis para que fiquem com
  tamanhos semelhantes.
  """
  len_pos = len(hard_positives)
  len_neg = len(hard_negatives)

  if len_pos > len_neg:
    max_size = int(len_neg)
    hard_positives = hard_positives.sample(n=max_size, random_state=42)
  elif len_neg > len_pos:
    max_size = int(len_pos)
    hard_negatives = hard_negatives.sample(n=max_size, random_state=42)

  return hard_positives, hard_negatives

# 4 - Separar a lista de positivos em 3 listas: treinamento, validação e teste, na proporção de 60%, 20% e 20% respectivamente, sendo que validação e teste devem ter o mesmo tamanho.
# 5 - Separar a lista de negativos em 3 listas: treinamento, validação e teste, na proporção de 60%, 20% e 20% respectivamente, sendo que validação e teste devem ter o mesmo tamanho.
# 6 - Unir as listas de treinamento (positivos e negativos), validação e teste, sendo que os pares devem estar dispostas em ordem aleatória.
# Ao final, serão três dataframas: dfTrain, dfValid e dfTest com o formato (ltable_id, rtable_id, label)

def split_train_valid_test(positives, negatives):
  """
  Separa as listas de positivos difíceis e negativos difíceis em treinamento, validação e teste.
  Retorna dataframes com os pares dispostos em ordem aleatória.
  """
  # Separa os positivos em treinamento, validação e teste
  pos_train, pos_valtest = train_test_split(positives, test_size=0.4, random_state=42, stratify=positives['label'])
  pos_val, pos_test = train_test_split(pos_valtest, test_size=0.5, random_state=42, stratify=pos_valtest['label'])

  # Separa os negativos em treinamento, validação e teste
  neg_train, neg_valtest = train_test_split(negatives, test_size=0.4, random_state=42, stratify=negatives['label'])
  neg_val, neg_test = train_test_split(neg_valtest, test_size=0.5, random_state=42, stratify=neg_valtest['label'])

  # Unifica as listas de treinamento, validação e teste
  train = pd.concat([pos_train, neg_train], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
  valid = pd.concat([pos_val, neg_val], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
  test = pd.concat([pos_test, neg_test], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

  return train[['ltable_id', 'rtable_id', 'label']], valid[['ltable_id', 'rtable_id', 'label']], test[['ltable_id', 'rtable_id', 'label']]

# Gerar exemplos para o treinamento e validação no formato do SentenceTransformers
def gera_input_examples_one_table(dfRotulos, dfTable):
  examples = []
  for d in dfRotulos.index:
    idLeft = dfRotulos['ltable_id'][d]
    idRight = dfRotulos['rtable_id'][d]
    currentSentenceA = dfTable['sentence'][idLeft]
    currentSentenceB = dfTable['sentence'][idRight]
    currentLabel = int(dfRotulos['label'][d])

    examples.append(InputExample(texts=[currentSentenceA, currentSentenceB], label=currentLabel))
  return examples

# Funções para etapa de junção

#Pegar valor do id do registro na tabela
def get_id(pos, df):
    return df.iloc[pos]['id']

# Auto junção com faiss
def auto_join(vetores,dfIds,tipo_indice,threshold):

  start = time.time()

  d = vetores.shape[1]

  if tipo_indice == 'IndexFlatIP':
    indice = faiss.IndexFlatIP(d)
  if tipo_indice == 'HNSW':
    #parameters hnsw
    M = 64
    efC = 32
    efS = 32
    indice = faiss.IndexHNSWFlat(d,M,faiss.METRIC_INNER_PRODUCT)

    indice.hnsw.efConstruction = efC
    indice.hnsw.efSearch = efS

  indice.add(vetores)

  end = time.time()

  timeIndex = end-start

  start = time.time()
  if tipo_indice =='IndexFlatIP':
    resultados = search_auto_IndexFlatIP(vetores,dfIds,indice,threshold)
  elif tipo_indice == 'HNSW':
    resultados = search_auto_HNSW(vetores,dfIds,indice,threshold)
  end = time.time()
  timeJoin = end-start

  return resultados,timeIndex,timeJoin


# Função de junção em duas tabelas com faiss
def join_tables(vetoresA,vetoresB,dfIdsA,dfIdsB,tipo_indice,threshold):

  if len(vetoresA) > len(vetoresB):
    invert = True
    vetoresIndexados = vetoresA
    idsIndex = dfIdsA
    vetoresBusca = vetoresB
    idsBusca = dfIdsB
  else:
    invert = False
    vetoresIndexados = vetoresB
    idsIndex = dfIdsB
    vetoresBusca = vetoresA
    idsBusca = dfIdsA

  start = time.time()

  d = vetoresIndexados.shape[1]

  if tipo_indice == 'IndexFlatIP':
    indice = faiss.IndexFlatIP(d)
  if tipo_indice == 'HNSW':
    #parameters
    M = 64
    efC = 32
    efS = 32
    indice = faiss.IndexHNSWFlat(d,M,faiss.METRIC_INNER_PRODUCT)

    indice.hnsw.efConstruction = efC
    indice.hnsw.efSearch = efS

  indice.add(vetoresIndexados)

  end = time.time()

  timeIndex = end-start

  start = time.time()
  if tipo_indice =='IndexFlatIP':
    resultados = search_IndexFlatIP(vetoresBusca,indice,idsBusca,idsIndex,invert,threshold)
  elif tipo_indice == 'HNSW':
    resultados = search_HNSW(vetoresBusca,indice,idsBusca,idsIndex,invert,threshold)
  end = time.time()
  timeJoin = end-start

  return resultados,timeIndex,timeJoin

# Função de junção em duas tabelas (somente em amostra dos dado rotulado)
def join_tables_sample(vetoresA,vetoresB,dfIdsA,dfIdsB,tipo_indice,threshold,sample,dfTest):

  # Gerar a amostra de dfTest com base no parâmetro sample
  dfTest_sample = dfTest.sample(frac=sample, random_state=42)  # Random state para reprodutibilidade

  # Filtrar os IDs de vetoresA e vetoresB com base na amostra gerada
  ids_ltable = set(dfTest_sample['ltable_id'])
  ids_rtable = set(dfTest_sample['rtable_id'])

  # Criar novos vetoresA e vetoresB contendo apenas os IDs da amostra
  maskA = [idx in ids_ltable for idx in dfIdsA['id']]
  maskB = [idx in ids_rtable for idx in dfIdsB['id']]

  vetoresA = vetoresA[maskA]
  vetoresB = vetoresB[maskB]
  dfIdsA = [idx for idx in dfIdsA['id'] if idx in ids_ltable]
  dfIdsB = [idx for idx in dfIdsB['id'] if idx in ids_rtable]
  dfIdsA = pd.DataFrame(dfIdsA, columns=['id'])
  dfIdsB = pd.DataFrame(dfIdsB, columns=['id'])


  if len(vetoresA) > len(vetoresB):
    invert = True
    vetoresIndexados = vetoresA
    idsIndex = dfIdsA
    vetoresBusca = vetoresB
    idsBusca = dfIdsB
  else:
    invert = False
    vetoresIndexados = vetoresB
    idsIndex = dfIdsB
    vetoresBusca = vetoresA
    idsBusca = dfIdsA

  start = time.time()

  d = vetoresIndexados.shape[1]

  if tipo_indice == 'IndexFlatIP':
    indice = faiss.IndexFlatIP(d)
  if tipo_indice == 'HNSW':
    #parameters
    M = 64
    efC = 32
    efS = 32
    indice = faiss.IndexHNSWFlat(d,M,faiss.METRIC_INNER_PRODUCT)

    indice.hnsw.efConstruction = efC
    indice.hnsw.efSearch = efS

  indice.add(vetoresIndexados)

  end = time.time()

  timeIndex = end-start

  start = time.time()
  if tipo_indice =='IndexFlatIP':
    resultados = search_IndexFlatIP(vetoresBusca,indice,idsBusca,idsIndex,invert,threshold)
  elif tipo_indice == 'HNSW':
    resultados = search_HNSW(vetoresBusca,indice,idsBusca,idsIndex,invert,threshold)
  end = time.time()
  timeJoin = end-start

  return resultados,timeIndex,timeJoin

def search_IndexFlatIP(vetoresBusca,indice,idsBusca,idsIndex,invert,threshold):
  resultados = []
  i = 0
  for vetor in vetoresBusca:
    vetor_consulta = vetor.reshape(1,-1).astype(np.float32)
    L, D, I = indice.range_search(vetor_consulta,thresh=threshold)
    if len(D) != 0:
      for result,prod in zip(I,D):
        if invert: resultados.append([result,i,idsIndex.id[result],idsBusca.id[i],prod])
        else: resultados.append([i,result,idsBusca.id[i],idsIndex.id[result],prod])
    i = i + 1
  return resultados

def search_auto_IndexFlatIP(vetores,dfIds,indice,threshold):
  resultados = []
  #i = vetores.shape[0] - 1 # em ordem reversa para não comparar o vetor com ele mesmo
  i = 0
  #for vetor in reversed(vetores):
  for vetor in vetores:
    #indice.remove_ids(np.array([i])) # remover do índice o vetor atual
    vetor_consulta = vetor.reshape(1,-1).astype(np.float32)
    L, D, I = indice.range_search(vetor_consulta,thresh=threshold)
    if len(D) != 0:
      for result,prod in zip(I,D):
        if result > i: # não incluir resultado com o próprio vetor e repetidos
          resultados.append([i,result,dfIds.id[i],dfIds.id[result],prod])
    i = i + 1
    #i = i - 1 # ordem reversa
  return resultados

def search_HNSW(vetoresBusca,indice,idsBusca,idsIndex,invert,threshold):
  resultados = []
  i = 0
  for vetor in vetoresBusca:
    vetor_consulta = vetor.reshape(1,-1).astype(np.float32)
    # heuristica para k inicial
    #topk = 10 # k inicial # version 1
    topk = int(round(32 + ((1 - threshold) * 32))) # version 2
    D, I = indice.search(vetor_consulta,k = topk)
    if len(D) != 0:
      if D.reshape(-1,1)[0] >= threshold:
        while D.reshape(-1,1)[-1] >= threshold:
          # heuristica para incremento do k
          #topk = 2 * topk # version 1
          #topk += int(round(((1 - threshold) * topk) / (1 - D.reshape(-1,1)[-1]))) # version 2
          topk += int(round(((1 - threshold) * topk) / (1 - D.reshape(-1)[-1].item())))

          D, I = indice.search(vetor_consulta,k = topk)
        for result,prod in zip(I.reshape(-1,1),D.reshape(-1,1)):
          if prod >= threshold:
            if invert: resultados.append([result[0],i,idsIndex.id[result[0]],idsBusca.id[i],prod[0]])
            else: resultados.append([i,result[0],idsBusca.id[i],idsIndex.id[result[0]],prod[0]])
    i = i + 1
  return resultados

def search_auto_HNSW(vetores,dfIds,indice,threshold):
  resultados = []
  i = 0
  for vetor in vetores:
    vetor_consulta = vetor.reshape(1,-1).astype(np.float32)
    # heuristica para k inicial
    #topk = 10 # k inicial # version 1
    topk = int(round(32 + ((1 - threshold) * 32))) # version 2
    D, I = indice.search(vetor_consulta,k = topk)
    if len(D) != 0:
      if D.reshape(-1,1)[0] >= threshold:
        while D.reshape(-1,1)[-1] >= threshold:
          # heuristica para incremento do k
          #topk = 2 * topk # version 1
          #topk += int(round(((1 - threshold) * topk) / (1 - D.reshape(-1,1)[-1]))) # version 2
          topk += int(round(((1 - threshold) * topk) / (1 - D.reshape(-1)[-1].item())))

          D, I = indice.search(vetor_consulta,k = topk)
        for result,prod in zip(I.reshape(-1,1),D.reshape(-1,1)):
          if prod >= threshold and result > i: # não incluir o vetor corrente no resultado e vetores abaixo do ths
            resultados.append( [i,result[0],dfIds.id[i],dfIds.id[result[0]],prod[0]])
    i = i + 1
  return resultados

# Função para calcular similaridade somente nos pares rotulados (tableA->tableB)
def join_test_two_tables(dfTest,vetoresA,vetoresB,dfIdsA,dfIdsB,threshold,sample):
  start = time.time()
  resultados = []

  if sample == 1:
    dfTest_sample = dfTest
  else:
    dfTest_sample = dfTest.sample(frac=sample, random_state=42)

  # Fazer loop para passar por todos os pares no dfTest
  for d in dfTest_sample.index:
    idLeft = dfTest_sample['ltable_id'][d]
    idRight = dfTest_sample['rtable_id'][d]

    # Pegar posição do ID de dfTest no dfIds
    if dfIdsA.iloc[0]['id'] != 0 and dfIdsA.iloc[(len(dfIdsA)-1)]['id'] != len(dfIdsA)-1:
      idLeft = dfIdsA[dfIdsA['id'] == idLeft].index[0]
      idRight = dfIdsB[dfIdsB['id'] == idRight].index[0]

    # Com a posição, pegar o par de vetores em vetoresA e vetoresB
    vetorA = vetoresA[idLeft].reshape(1, -1)
    vetorB = vetoresB[idRight].reshape(1, -1)

    # Calcular a similaridade de cosseno entre o par de vetores
    sim = cosine_similarity(vetorA,vetorB)

    # Se a similaridade for maior que o threshold, incluir na lista de resultados
    if sim > threshold:
      resultados.append([idLeft,idRight,dfIdsA.id[idLeft],dfIdsB.id[idRight],sim])
  end = time.time()
  timeJoin = end - start
  # retornar a lista de resultados
  return resultados,timeJoin

# Função para calcular similaridade somente nos pares rotulados (única tabela)
def join_test_one_table(dfTest,vetores,dfIds,threshold):
  start = time.time()
  resultados = []
  # Fazer loop para passar por todos os pares no dfTest
  for d in dfTest.index:
    idLeft = dfTest['ltable_id'][d]
    idRight = dfTest['rtable_id'][d]

    # Com a posição, pegar o par de vetores
    vetorLeft = vetores[idLeft].reshape(1, -1)
    vetorRight = vetores[idRight].reshape(1, -1)

    # Calcular a similaridade de cosseno entre o par de vetores
    sim = cosine_similarity(vetorLeft,vetorRight)

    # Se a similaridade for maior que o threshold, incluir na lista de resultados
    if sim > threshold:
      resultados.append([idLeft,idRight,dfIds.id[idLeft],dfIds.id[idRight],sim])
  end = time.time()
  timeJoin = end - start
  # retornar a lista de resultados
  return resultados,timeJoin

# depracated (substituído pelo método com uso do sklearn)
def exibir_metricas(resultados,dfRotulos,dfIdsA,dfIdsB):
  dfR = pd.DataFrame(resultados)
  total_recuperados = 0
  verdadeiros = 0
  falsos = 0
  lista_verdadeiros = []
  lista_falsos = []

  for lid,rid in zip(dfR[0].values,dfR[1].values):
    left =  dfIdsA.iloc[lid][0]
    right =  dfIdsB.iloc[rid][0]

    #verificar se resultado consta como registro ns rótulos, se não passar para o próximo
    if (left,right) not in zip(dfRotulos['ltable_id'],dfRotulos['rtable_id']):
      continue
    else:
      total_recuperados += 1
      # veriricar se registro par está como positivo ou negativo
      if (left,right,1) in zip(dfRotulos['ltable_id'],dfRotulos['rtable_id'],dfRotulos['label']):
        verdadeiros += 1
        lista_verdadeiros.append([left,right])
      else:
        falsos +=1
        lista_falsos.append([left,right])

  # Indicadores do resultado
  candidatos_total = len(dfRotulos)
  pares_similares_total = (dfRotulos['label'] == 1).sum()

  acuracia = (verdadeiros + (candidatos_total - total_recuperados)) / candidatos_total
  precisao = verdadeiros / total_recuperados
  recall = verdadeiros / pares_similares_total
  f1score =  (2 * precisao * recall) / (precisao + recall )

  print('Total pares rotulados:',candidatos_total)
  print('Total rótulos positivos:',pares_similares_total)
  print('Pares recuperados dentro dos rótulos:',total_recuperados)
  print('Verdadeiros:',verdadeiros)
  print('Falsos:',falsos)
  print('Acurácia: {:.7f}%'.format(acuracia*100))
  print('Precisão: {:.3f}%'.format(precisao*100))
  print('Recall: {:.3f}%'.format(recall*100))
  print('F1-score: {:.3f}%'.format(f1score*100))

# depracated (substituído pelo método com uso do sklearn)
def get_precison_recall_f1_verdadeiros_falsos(resultados,dfRotulos):
  dfR = pd.DataFrame(resultados)
  total_recuperados = 0
  verdadeiros = 0
  falsos = 0
  lista_verdadeiros = []
  lista_falsos = []
  for left,right in zip(dfR[0].values,dfR[1].values):
    #verificar se resultado consta como registro no teste, se não passar para o próximo
    if (left,right) not in zip(dfRotulos['ltable_id'],dfRotulos['rtable_id']):
      continue
    else:
      total_recuperados += 1
      # veriricar se registro par está como positivo ou negativo
      if (left,right,1) in zip(dfRotulos['ltable_id'],dfRotulos['rtable_id'],dfRotulos['label']):
        verdadeiros += 1
        lista_verdadeiros.append([left,right])
      else:
        falsos +=1
        lista_falsos.append([left,right])
  # Indicadores do resultado
  candidatos_total = len(dfRotulos)
  pares_similares_total = (dfRotulos['label'] == 1).sum()

  precisao = verdadeiros / total_recuperados
  recall = verdadeiros / pares_similares_total
  f1score =  (2 * precisao * recall) / (precisao + recall )

  return precisao,recall,f1score,lista_verdadeiros,lista_falsos

# depracated (substituído pelo método com uso do sklearn)
def encontrar_melhor_threshold(vetoresA,vestoresB,index,threshold,dfRotulos):
  d = vetoresB.shape[1]
  if index == 'IndexFlatIP':
    indice = faiss.IndexFlatIP(d)
  indice.add(vetoresB)

  #selecionar somente os vetores contidos no dfRotulos['ltable_id']
  #ids_left = dfRotulos['ltable_id'].values
  #vetoresLeft = vetoresA[ids_left]

  def search_index(threshold):
    resultados = []
    i = 0
    #for vetor in vetoresLeft:
    for vetor in vetoresA:
      vetor_consulta = vetor.reshape(1,-1).astype(np.float32)
      L, D, I = indice.range_search(vetor_consulta,thresh=threshold)
      if len(D) != 0:
        for result,prod in zip(I,D):
          resultados.append([i,result,prod])
          #resultados.append([ids_left[i],result,prod]) #ids_lefts ao invés de i
      i = i + 1
    return resultados

  best_threshold = 0.0
  best_f1 = 0.0
  current_threshold = 0.8
  while True:
    resultados = search_index(current_threshold)
    precision,recall,f1,verdadeiros,falsos = get_precison_recall_f1_verdadeiros_falsos(resultados,dfValid)

    # atualizar o melhor threshold se o F1-Score atual for melhor que o anterior
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = current_threshold
    # parar a busca se o F1-Score atual for pior que o anterior
    elif f1 < best_f1:
        break

    # atualizar o threshold atual para a próxima iteração
    current_threshold -= 0.05
    current_threshold = round(current_threshold, 2)
  return best_threshold

# Método para exibir as métricas (junção tableA->tableB) com uso da biblioteca sklearn
def get_metrics_sklearn_two_tables(resultados,dfRotulos):

  dfR = pd.DataFrame(resultados)
  dfR.columns = ['pos_l', 'pos_r', 'ltable_id', 'rtable_id', 'predicted_label']
  dfR['predicted_label'] = 1

  #Substituir a posição pelo valor do ID, caso o ID não seja a posição do registro (ex: dataset company)
  #if dfIdsA.iloc[0]['id'] != 0 and dfIdsA.iloc[(len(dfIdsA)-1)]['id'] != len(dfIdsA)-1:
  #  dfR['ltable_id'] = dfR['ltable_id'].apply(get_id, args=(dfIdsA,))
  #  dfR['rtable_id'] = dfR['rtable_id'].apply(get_id, args=(dfIdsB,))

  # Junta os dois dataframes usando as colunas "ltable_id" e "rtable_id" #left, outer, right
  df = pd.merge(dfRotulos, dfR, on=['ltable_id', 'rtable_id'], how='left')

  #Completa de zero demais predicted_labels (pares que não vieram no resultado e constam nos Rotulos)
  df = df.fillna(0)

  # Calcule as métricas
  metrics = precision_recall_fscore_support(df['label'], df['predicted_label'], pos_label=1, average='binary')
  log_m = ""
  log_m += f"Precisão: {metrics[0]*100:.2f}\n"
  log_m += f"Recall: {metrics[1]*100:.2f}\n"
  log_m += f"F1-Score: {metrics[2]*100:.2f}"

  return log_m

# Método para exibir as métricas (auto-junção) junção com todos os registros
def get_metrics_auto_join_full(resultados,dfIds):

  dfR = pd.DataFrame(resultados)
  dfR.columns = ['pos_l', 'pos_r', 'id_l', 'id_r', 'score']
  #dfR['predicted_label'] = 1

  # total de recuperados
  total_recuperados = len(dfR)

  # encontrar os pares verdadeiros no resultado
  verdadeiros = (dfR['id_l'] == dfR['id_r']).sum()

  # encontrar os pares falsos
  falsos = total_recuperados - verdadeiros

  # identificar quais registro possuem duplicatas e quantas
  duplicatas = []
  unicos = set(dfIds.id)
  vc = dfIds['id'].value_counts()
  duplicatas = vc[vc > 1].index.tolist()

  # encontrar a quantidade total de pares positivos Combinação de(n,k)
  total_combinacoes = 0
  for valor in duplicatas:
    count = vc[valor]
    total_combinacoes += math.comb(count, 2)
  pares_similares_total = total_combinacoes

  # Indicadores do resultado

  precisao = verdadeiros / total_recuperados
  recall = verdadeiros / pares_similares_total
  if precisao == 0 and recall == 0:
    f1score = 0
  else:
    f1score =  (2 * precisao * recall) / (precisao + recall )

  #print('total recuperados:',total_recuperados)
  #print('verdadeiros:',verdadeiros)
  #print('falsos:',falsos)
  #print('pares similares total',pares_similares_total)

  metrics = ""
  metrics += f"Precisão: {precisao*100:.2f}\n"
  metrics += f"Recall: {recall*100:.2f}\n"
  metrics += f"F1-Score: {f1score*100:.2f}"

  return metrics

# Método para exibir as métricas (auto-junção) junção somente nos dados de teste
def get_metrics_auto_join_test(resultados,dfTest):

  dfR = pd.DataFrame(resultados)
  dfR.columns = ['pos_l', 'pos_r', 'id_l', 'id_r', 'score']

  # total de recuperados
  total_recuperados = len(dfR)

  # encontrar os pares verdadeiros no resultado
  verdadeiros = (dfR['id_l'] == dfR['id_r']).sum()

  # encontrar os pares falsos
  falsos = total_recuperados - verdadeiros

  # encontrar a quantidade total de pares positivos no dfTest
  pares_similares_total = (dfTest['label'] == 1).sum()

  # Indicadores do resultado
  precisao = verdadeiros / total_recuperados
  recall = verdadeiros / pares_similares_total
  if precisao == 0 and recall == 0:
    f1score = 0
  else:
    f1score =  (2 * precisao * recall) / (precisao + recall )

  metrics = ""
  metrics += f"Precisão: {precisao*100:.2f}\n"
  metrics += f"Recall: {recall*100:.2f}\n"
  metrics += f"F1-Score: {f1score*100:.2f}"

  return metrics

#Método para encontrar o melhor threshold (que gera maior valor de F1) (tableA->tableB)
def find_best_threshold_two_tables(resultados, dfTest):

  dfR = pd.DataFrame(resultados)
  dfR.columns = ['pos_l', 'pos_r', 'ltable_id', 'rtable_id', 'score']

  #Substituir a posição pelo valor do ID, caso o ID não seja a posição do registro (ex: dataset company)
  #if dfIdsA.iloc[0]['id'] != 0 and dfIdsA.iloc[(len(dfIdsA)-1)]['id'] != len(dfIdsA)-1:
  #  dfR['ltable_id'] = dfR['ltable_id'].apply(get_id, args=(dfIdsA,))
  #  dfR['rtable_id'] = dfR['rtable_id'].apply(get_id, args=(dfIdsB,))

  # Ordena os resultados em ordem decrescente de score
  dfR = dfR.sort_values(by='score', ascending=False)

  best_f1 = 0.0
  best_threshold = 0.95
  current_threshold = 0.95

  while True:
    # Filtra os resultados de acordo com o limiar
    filtered_results = dfR[dfR['score'] >= current_threshold]

    filtered_results = filtered_results.rename(columns={'score': 'predicted_label'})
    filtered_results['predicted_label'] = 1

    # Junta os dois dataframes usando as colunas "ltable_id" e "rtable_id"
    df = pd.merge(dfTest, filtered_results, on=['ltable_id', 'rtable_id'], how='left')

    # Substitui os valores nulos por 0
    df = df.fillna(0)

    # Calcula as métricas
    metrics = precision_recall_fscore_support(df['label'], df['predicted_label'], pos_label=1, average='binary',zero_division=1)
    f1 = metrics[2]

    # Verifica se é o melhor limiar até agora
    if f1 > best_f1:
      best_f1 = f1
      best_threshold = current_threshold
      current_threshold -= 0.05
      current_threshold = round(current_threshold,2)
    else:
      break
  delta_threshold = np.arange(best_threshold - 0.1, best_threshold + 0.05, 0.01)

  for ths in delta_threshold:
    ths = round(ths,2)
    filtered_results = dfR[dfR['score'] >= ths]
    filtered_results = filtered_results.rename(columns={'score': 'predicted_label'})
    filtered_results['predicted_label'] = 1

    df = pd.merge(dfTest, filtered_results, on=['ltable_id', 'rtable_id'], how='left')
    df = df.fillna(0)

    metrics = precision_recall_fscore_support(df['label'], df['predicted_label'], pos_label=1, average='binary', zero_division=1)
    f1 = metrics[2]

    # Verifica se o novo f1 é melhor que o anterior
    if f1 > best_f1:
      best_f1 = f1
      best_threshold = ths

  return best_threshold,best_f1

# Método para obter F1 nos resultados de auto junção
def get_f1_auto_join(results,pares_similares_total):

  # total de recuperados
  total_recuperados = len(results)

  # encontrar os pares verdadeiros no resultado
  verdadeiros = (results['ltable_id'] == results['rtable_id']).sum()

  # encontrar os pares falsos
  falsos = total_recuperados - verdadeiros

  # Indicadores do resultado
  precisao = verdadeiros / total_recuperados
  recall = verdadeiros / pares_similares_total
  if precisao == 0 and recall == 0:
    f1score = 0
  else:
    f1score =  (2 * precisao * recall) / (precisao + recall )
    f1score = f1score * 100

  return f1score

#Método para encontrar o melhor threshold (auto-junção) para junção completa
def find_best_threshold_auto_join(resultados, dfIds):

  dfR = pd.DataFrame(resultados)
  dfR.columns = ['pos_l', 'pos_r', 'ltable_id', 'rtable_id', 'score']

  # Ordena os resultados em ordem decrescente de score
  dfR = dfR.sort_values(by='score', ascending=False)

  # identificar quais registro possuem duplicatas e quantas
  duplicatas = []
  unicos = set(dfIds.id)
  vc = dfIds['id'].value_counts()
  duplicatas = vc[vc > 1].index.tolist()

  # encontrar a quantidade total de pares positivos Combinação de(n,k)
  total_combinacoes = 0
  for valor in duplicatas:
    count = vc[valor]
    total_combinacoes += math.comb(count, 2)
  pares_similares_total = total_combinacoes

  best_f1 = 0.0
  best_threshold = 0.95
  current_threshold = 0.95

  while True:
    # Filtra os resultados de acordo com o limiar
    filtered_results = dfR[dfR['score'] >= current_threshold]

    # Obtém F1
    f1 = get_f1_auto_join(filtered_results,pares_similares_total)

    # Verifica se é o melhor limiar até agora
    if f1 > best_f1:
      best_f1 = f1
      best_threshold = current_threshold
      current_threshold -= 0.05
      current_threshold = round(current_threshold,2)
    else:
      break
  delta_threshold = np.arange(best_threshold - 0.1, best_threshold + 0.05, 0.01)

  for ths in delta_threshold:
    ths = round(ths,2)
    filtered_results = dfR[dfR['score'] >= ths]

    # Obtém F1
    f1 = get_f1_auto_join(filtered_results,pares_similares_total)

    # Verifica se o novo f1 é melhor que o anterior
    if f1 > best_f1:
      best_f1 = f1
      best_threshold = ths

  return best_threshold,best_f1


#Método para encontrar o melhor threshold (auto-junção) para junção somente nos dados de teste
def find_best_threshold_auto_join_test(resultados, dfTest):

  dfR = pd.DataFrame(resultados)
  dfR.columns = ['pos_l', 'pos_r', 'ltable_id', 'rtable_id', 'score']

  # Ordena os resultados em ordem decrescente de score
  dfR = dfR.sort_values(by='score', ascending=False)

  # encontrar a quantidade total de pares positivos no dfTest
  pares_similares_total = (dfTest['label'] == 1).sum()

  best_f1 = 0.0
  best_threshold = 0.95
  current_threshold = 0.95

  while True:
    # Filtra os resultados de acordo com o limiar
    filtered_results = dfR[dfR['score'] >= current_threshold]

    # Obtém F1
    f1 = get_f1_auto_join(filtered_results,pares_similares_total)

    # Verifica se é o melhor limiar até agora
    if f1 > best_f1:
      best_f1 = f1
      best_threshold = current_threshold
      current_threshold -= 0.05
      current_threshold = round(current_threshold,2)
    else:
      break
  delta_threshold = np.arange(best_threshold - 0.1, best_threshold + 0.05, 0.01)

  for ths in delta_threshold:
    ths = round(ths,2)
    filtered_results = dfR[dfR['score'] >= ths]

    # Obtém F1
    f1 = get_f1_auto_join(filtered_results,pares_similares_total)

    # Verifica se o novo f1 é melhor que o anterior
    if f1 > best_f1:
      best_f1 = f1
      best_threshold = ths

  return best_threshold,best_f1

# PIPELINE EXEC
startPip = time.time()
log = []
log.append('---- Informações da geração do modelo ----')

import sys, os, subprocess
import argparse

base_path = os.getcwd() + '/datasets/'

# Criação do parser de argumentos
parser = argparse.ArgumentParser(description='Configurações de Treinamento e Processamento')

# Parâmetros do dataset
parser.add_argument('--collection', type=str, default='sparkly-all', help='Coleção do dataset')
parser.add_argument('--type_dataset', type=str, default='magellan', help='Tipo do dataset')
parser.add_argument('--dataset', type=str, default='song-song', help='Nome do dataset')
parser.add_argument('--device', type=str, default='cuda', help='Dispositivo de execução')

# Parâmetros de pré-processamento
parser.add_argument('--token_separator', type=str, default=' [SEP] ', choices=[' ', ' [SEP] ', ' [CLS] ', ' [PAD] '], help='Separador entre tokens')
parser.add_argument('--NaN_substitute', type=str, default=' ', choices=[' ', '0', ' [BLANK] ', ' [MISSING] '], help='Substituto para valores NaN')
parser.add_argument('--remove_numbers', action='store_true', help='Remover números do texto')
parser.add_argument('--remove_special_chars', action='store_true', help='Remover caracteres especiais')
parser.add_argument('--stem_sentence', action='store_true', help='Aplicar stemização')
parser.add_argument('--lemmatize_sentence', action='store_true', help='Aplicar lematização')
parser.add_argument('--summarize_sentence', action='store_true', help='Aplicar sumarização')
parser.add_argument('--percent_summarize', type=float, default=0.02, help='Percentual de sumarização para registros com muitas palavras')
parser.add_argument('--data_augmentation', action='store_true', help='Ativar reconhecimento de entidades e incluir nome antes do texto')

# Parâmetros do modelo
parser.add_argument('--model_pre', type=str, default='all-MiniLM-L12-v2', help='Modelo pré-treinado para embeddings')
parser.add_argument('--normalize_embedings', action='store_true', help='Normalizar embeddings')
parser.add_argument('--load_local_model_save', action='store_true', help='Carregar modelo salvo localmente')
parser.add_argument('--generate_train_data', action='store_true', help='Gerar dados de treinamento')
parser.add_argument('--save_train_data', action='store_true', help='Salvar dados de treinamento')
parser.add_argument('--load_train_data', action='store_true', help='Carregar dados de treinamento salvos')
parser.add_argument('--perform_fine_tuning', action='store_true', help='Realizar fine-tuning')
parser.add_argument('--only_positive_labels', action='store_true', help='Usar apenas rótulos positivos')

# Parâmetros de treinamento
parser.add_argument('--num_epochs', type=int, default=40, help='Número de épocas de treinamento')
parser.add_argument('--batch_size', type=int, default=8, help='Tamanho do batch')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Taxa de aprendizado')
parser.add_argument('--scheduler', type=str, default='ConstantLR', choices=['ConstantLR', 'WarmupConstant', 'WarmupLinear', 'WarmupCosine', 'WarmupCosineWithHardRestarts'], help='Scheduler de aprendizado')
parser.add_argument('--save_best_model_ft', action='store_true', help='Salvar o melhor modelo de fine-tuning')
parser.add_argument('--overwrite_exist_model', action='store_true', help='Sobrescrever modelo existente')
parser.add_argument('--is_model_tuned', action='store_true', help='Define se o modelo foi ajustado')


# Parâmetros para geração de vetores
parser.add_argument('--generate_vectors', action='store_true', help='Gerar vetores de dados')
parser.add_argument('--save_vectors', action='store_true', help='Salvar vetores gerados')
parser.add_argument('--load_save_vectors', action='store_true', help='Carregar vetores salvos')

# Parâmetros para a etapa de junção
parser.add_argument('--perform_join', action='store_true', help='Realizar junção de registros')
parser.add_argument('--threshold', type=float, default=0.6, help='Valor de threshold definido manualmente ou threshold inicial para o auto-threshold')
parser.add_argument('--auto_threshold', action='store_true', help='Definir limite automaticamente')
parser.add_argument('--data_use', type=str, default='train+valid', choices=['train', 'valid', 'train+valid', 'test'], help='Dados rotulados que serão utilizado na busca do auto threshold')
parser.add_argument('--join_sample', action='store_true', help='Utilizar amostra dos dados rotulados para auto threshold')
parser.add_argument('--sample', type=float, default=1, help='Valor da amostra 1=100%')
parser.add_argument('--index', type=str, default='IndexFlatIP', choices=['IndexFlatIP', 'HNSW', 'PairsInTestOnly'], help='Índice de similaridade')
parser.add_argument('--show_metrics', action='store_true', help='Exibir métricas')
parser.add_argument('--save_log_metrics', action='store_true', help='Salvar log e métricas')
parser.add_argument('--save_result', action='store_true', help='Salvar resultado final')

# Parse dos argumentos
args = parser.parse_args()

# Exemplo de uso dos parâmetros
print(f'Coleção: {args.collection}')
print(f'Tipo do dataset: {args.type_dataset}')
print(f'Dataset: {args.dataset if args.collection in ["deepmatcher", "sparkly-all"] else ""}')
print(f'Dispositivo: {args.device}')
print(f'Separador de tokens: {args.token_separator}')
print(f'Normalizar embeddings: {args.normalize_embedings}')
print(f'Número de épocas: {args.num_epochs}')
print(f'Taxa de aprendizado: {args.learning_rate}')
print(f'Scheduler: {args.scheduler}')

collection = args.collection
type_dataset = args.type_dataset
dataset = args.dataset
if collection != 'deepmatcher' and collection != 'sparkly-all':
  dataset = ''
device = args.device

# 1. PRÉ-PROCESSAMENTO
token_separator = args.token_separator
NaN_substitute = args.NaN_substitute
remove_numbers = args.remove_numbers
remove_special_chars = args.remove_special_chars
stem_sentence = args.stem_sentence
lemmatize_sentence = args.lemmatize_sentence
summarize_sentence = args.summarize_sentence
percent_summarize = args.percent_summarize
data_augmentation = args.data_augmentation

# 2. REPRESENTAÇÃO DOS DADOS
model_pre = args.model_pre
normalize_embedings = args.normalize_embedings
load_local_model_save = args.load_local_model_save

generate_train_data = args.generate_train_data
save_train_data = args.save_train_data
load_train_data = args.load_train_data

perform_fine_tuning = args.perform_fine_tuning
only_positive_labels = args.only_positive_labels

# Inserir Parâmetros de treinamento
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
scheduler = args.scheduler

save_best_model_ft = args.save_best_model_ft
overwrite_exist_model = args.overwrite_exist_model

is_model_tuned = 'True' if load_local_model_save or perform_fine_tuning else 'False'

# Gerar vetores com modelo selecionado
generate_vectors = args.generate_vectors
save_vectors = args.save_vectors
load_save_vectors = args.load_save_vectors

# 3. JUNÇÃO
perform_join = args.perform_join
threshold = args.threshold
auto_threshold = args.auto_threshold
data_use = args.data_use
join_sample = args.join_sample
sample = args.sample
index = args.index
show_metrics = args.show_metrics
save_log_metrics = args.save_log_metrics
save_result = args.save_result

log.append('- dataset: '+collection +' '+ type_dataset +' '+dataset)
log.append('- model: '+model_pre)
log.append('- fine-tuning in %d epochs and %d batch_sizes:' % (num_epochs,batch_size))
log.append('- token_separator: '+token_separator)
log.append('- NaN_substitute: '+NaN_substitute)

#### Acessar arquivos
if collection == 'deepmatcher' or collection == 'sparkly-all':
  path_full = str(base_path) + str(collection) + '/' + str(type_dataset) + '/' + str(dataset) + '/'
  csv_tableA = path_full + 'tableA.csv'
  csv_tableB = path_full + 'tableB.csv'
  csv_train = path_full + 'train.csv'
  csv_valid = path_full + 'valid.csv'
  csv_test = path_full + 'test.csv'
else:
  path_full = str(base_path) + str(collection) + '/' + str(type_dataset) + '/'
  tsv_table = path_full + 'table.tsv'
  csv_train = path_full + 'train.csv'
  csv_valid = path_full + 'valid.csv'
  csv_test = path_full + 'test.csv'

path_save_model = path_full + 'models/' + str(model_pre.replace("/", "-")) + '_' + str(num_epochs) +'_'+str(batch_size)+'/'


# Carregar dados nos respectivos dataframes e realizando pré-processamento nos dados
print('1. PRÉ-PROCESSAMENTO')
print('-> Carregando dados do dataset',collection,type_dataset,dataset)
start = time.time()
if collection == 'deepmatcher' or collection == 'sparkly-all':
  dfIdsA,dfIdsB,dfA,dfB,dfTrain,dfValid,dfTest = process_tables_labels(csv_tableA,csv_tableB,csv_train,csv_valid,csv_test,token_separator,NaN_substitute,remove_numbers,remove_special_chars,stem_sentence,lemmatize_sentence,summarize_sentence,percent_summarize,data_augmentation)
else:
  dfIds,dfTable = process_table(tsv_table,token_separator,NaN_substitute,remove_numbers,remove_special_chars,stem_sentence,lemmatize_sentence,summarize_sentence,percent_summarize,data_augmentation)
end = time.time()
print('Tempo para carregar e pré-processar dados: %2.fms' % ((end-start)*1000))



print('\n2. REPRESENTAÇÃO DOS DADOS')

class SaveModelException(Exception):
  pass

##### Carregar modelo SentenceTransformers conforme parâmetros
if (load_local_model_save):
  try:
    print('-> Carregando modelo do drive')
    model = SentenceTransformer(path_save_model,device=device)
    print('-> Modelo carregado com sucesso: ',path_save_model)

    ## Teste de salvamento
    if save_best_model_ft and not overwrite_exist_model:
      print('ATENÇÃO: Salvamento automático marcado. Para sobrescrever o modelo salvo, marque a opção overwrite_exist_model! Execução interrompida')
      raise SaveModelException()
  except SaveModelException:
    sys.exit()
  except Exception:
    print('-> ATENÇÃO: O modelo solicitado não possui versão com fine-tuning salva no drive! Execução interrompida.')
    sys.exit()
else:
  print('-> Carregando modelo do HuggingFace')
  #trecho adicionado para setella
  if model_pre == "dunzhang/stella_en_400M_v5":
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
  elif model_pre == "Alibaba-NLP/gte-large-en-v1.5":
    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True).cuda()
  else:
    model = SentenceTransformer(model_pre,device=device)

## Geração de dados de treinamento, validação e teste
if generate_train_data:
  start = time.time()
  print('-> Iniciando geração de dados de treinamento, validação e teste')

  if collection == 'deepmatcher' or collection == 'sparkly-all':
    print('Geração de dados de treinamento para duas tabelas não implemementado.')
  else:
    print('Gerando vetores dos dados do dataset',collection,type_dataset)
    vetores = gera_vetores_one_table(model,dfTable,normalize_embedings)

    print('Executando junção para gerar amostra de pares e suas similaridades')
    dfResults = find_similar_pairs(vetores, dfIds, k=20)

    ### Somente pares difíceis
    #print('Gerando pares positivos e negativos dificeis')
    #hard_pos,hard_neg = generate_hard_labels(dfResults, pos_threshold=0.6,neg_threshold=0.8)
    #print('Realizando balanceamento dos dados positivos e negativos')
    #labels_pos,labels_neg = balance_pos_neg(hard_pos, hard_neg)

    print('Gerando pares rotulados com balanceamento entre positivos e negativos, fáceis e difíceis.')
    labels_pos, labels_neg = generate_balanced_labels(dfResults, pos_threshold=0.6,neg_threshold=0.8)

    print('Gerando dados de treinamento, validação e teste')
    dfTrain,dfValid,dfTest = split_train_valid_test(labels_pos, labels_neg)

    print('Dados de treinamento (dfTrain), validação (dfValid) e teste (dfTest) gerados com sucesso.')
    end = time.time()
    print('Tempo para gerar dados: %2.fs \n' % (end-start))

  if save_train_data:
    print('Salvando dados de treinamento, validação e teste em arquivos...')
    url_train = path_full + 'train.csv'
    url_valid = path_full + 'valid.csv'
    url_test = path_full + 'test.csv'
    dfTrain.to_csv(url_train, header=True, index=False)
    dfValid.to_csv(url_valid, header=True, index=False)
    dfTest.to_csv(url_test, header=True, index=False)
    print('Dados salvos com sucesso:')
    print('dfTrain:',url_train)
    print('dfValid',url_valid)
    print('dfTest',url_test)

if load_train_data:
  print('Carregado dados de treinamento, validação e teste do arquivo.')
  dfTrain = pd.read_csv(csv_train,sep=',',header = 0)
  dfValid = pd.read_csv(csv_valid,sep=',',header = 0)
  dfTest = pd.read_csv(csv_test,sep=',',header = 0)
  print('Dados carregados com sucesso.')


#### Realiza fine-tuning caso True
if (perform_fine_tuning):
  print('-> Iniciando processo de fine-tuning')
  # Carregar dados de treinamento

  print('-> Carregando dados de treinamento')
  if only_positive_labels:
    print('-> Carregando somente rótulos positivos')
    dfTrainModel = dfTrain.loc[dfTrain['label'] == 1]
  else:
    dfTrainModel = dfTrain

  # Gera exemplos de treinamento
  if collection == 'deepmatcher' or collection == 'sparkly-all':
    train_examples = gera_input_examples(dfTrainModel,dfA,dfB,dfIdsA,dfIdsB)
  else:
    train_examples = gera_input_examples_one_table(dfTrain, dfTable)

  # Carregar dados de avaliação
  print('-> Carregando dados de avaliação')
  if collection == 'deepmatcher' or collection == 'sparkly-all':
    valid_examples = gera_input_examples(dfValid,dfA,dfB,dfIdsA,dfIdsB)
  else:
    valid_examples = gera_input_examples_one_table(dfValid, dfTable)

  # Definir evaluator
  evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(valid_examples, name='valid_examples')

  # Definindo o dataloader e parâmetros de treinamento
  train_batch_size = batch_size
  train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
  steps_per_epoch = len(train_dataloader) // train_batch_size
  num_evaluator_steps = len(train_dataloader)
  num_total_steps = steps_per_epoch * num_epochs
  warmup_steps = num_total_steps

  # Definindo a função de perda

  #ConstrastiveLoss
  train_loss = losses.ContrastiveLoss(model=model)

  #OnlineConstrativeLoss
  #margin = 0.5
  #distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
  #train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

  log.append('- learning_rate: '+str(learning_rate))
  log.append('- scheduler: '+scheduler)
  log.append('- batch_size: '+str(batch_size))
  log.append('- epochs: '+str(num_epochs))
  log.append('- total_steps: '+str(num_total_steps))

  if (not save_best_model_ft):
    if load_local_model_save:
      print('-> Fine-tuning presente no drive para os parâmetros informados! Modelo carregado do drive para inciar fine-tuning a partir do mesmo.')
    else:
      print('-> Ininiciar fine-tuning a partir de modelo carregado do HuggingFace!')
    print('-> Iniciando treinamento do modelo SEM salvamento automático! Salvar manualmente modelo e log caso necessário.')
    start = time.time()
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              steps_per_epoch=steps_per_epoch,
              evaluation_steps = num_evaluator_steps,
              scheduler = scheduler,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': learning_rate},
              show_progress_bar=True)
    end = time.time()
    log.append('- Tempo para fine-tuning: %2.fs \n' % (end-start))
    output_gpu = subprocess.check_output(['nvidia-smi'])
    gpu_info = output_gpu.decode('utf-8')
    log.append(gpu_info)
    log = "\n".join(log)
    print(log)
  else:
    if load_local_model_save:
      print('-> Fine-tuning presente no drive para os parâmetros informados! Modelo carregado do drive para inciar fine-tuning a partir do mesmo.')
      if not overwrite_exist_model:
        print('ATENÇÃO: Para salvar o treinamento, marque a opção overwrite_exist_model')
        print('Execução interrompida!')
        sys.exit()
      else:
        print('-> Iniciando treinamento do modelo COM salvamento automático sobrescrevendo modelo já salvo!')
    else:
      print('-> Verificando se modelo já salvo no drive com os parâmetros de fine-tuning informados.')
      try:
        model_test = SentenceTransformer(path_save_model,device=device)
        print('-> O modelo possui versão com fine-tuning salva no drive. Caso deseje utilizá-lo, marcar a opção load_local_model_save')
        if not overwrite_exist_model:
          print('ATENÇÃO: Para salvar o treinamento, marque a opção overwrite_exist_model')
          print('Execução interrompida!')
          raise SaveModelException()
        else:
          print('-> Iniciando treinamento do modelo COM salvamento automático sobrescrevendo modelo já salvo!')
      except SaveModelException:
        sys.exit()
      except:
        print('-> O modelo solicitado não possui versão com fine-tuning salva no drive!')
        print('-> Iniciar fine-tuning a partir de modelo carregado do HuggingFace!')
        print('-> Iniciando treinamento do modelo COM salvamento automático!')

    start = time.time()
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              steps_per_epoch=steps_per_epoch,
              evaluation_steps = num_evaluator_steps,
              show_progress_bar=True,
              save_best_model=True,
              scheduler = scheduler,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': learning_rate},
              output_path = path_save_model)
    end = time.time()
    log.append('- Tempo para fine-tuning: %2.fs \n' % (end-start))
    url_log = path_save_model + 'log_fine-tuning.txt'
    output_gpu = subprocess.check_output(['nvidia-smi'])
    gpu_info = output_gpu.decode('utf-8')
    log.append(gpu_info)
    log = "\n".join(log)
    print(log)
    with open(url_log, 'w', encoding='utf-8') as f:
      f.write(log.encode('utf-8').decode('utf-8'))
    print('-> Carregando melhor modelo')
    #trecho adicionado para setella
  if model_pre == "dunzhang/stella_en_400M_v5":
    model = SentenceTransformer(path_save_model, trust_remote_code=True).cuda()
  elif model_pre == "Alibaba-NLP/gte-large-en-v1.5":
    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True).cuda()
  else:
    model = SentenceTransformer(path_save_model,device=device)

if generate_vectors:
  start = time.time()
  if collection == 'deepmatcher' or collection == 'sparkly-all':
    print('-> Iniciando geração de vetores das sentenças de tableA e tableB. Normalizar:',normalize_embedings,'Se falso, confirmar se modelo já normaliza por default.')
    vetoresA,vetoresB = gera_vetores(dfA['sentence'],dfB['sentence'],model,normalize_embedings)
    print('Linhas/Dimensões - vetoresA:',vetoresA.shape,'vetoresB:',vetoresB.shape)
  else:
    print('-> Iniciando geração de vetores das sentenças do dataset',collection,type_dataset)
    print('Normalizar embeddings:',normalize_embedings)
    vetores = model.encode(dfTable['sentence'],show_progress_bar=True,normalize_embeddings=normalize_embedings)
    print('Linhas/Dimensões:',vetores.shape)
  end = time.time()
  print('Tempo para encodding: %2.fms' % ((end-start)*1000))

  if save_vectors:
    print('-> Salvando vetores em arquivo...')
    dir_save_vectors = path_full + 'vectors/'+ str(model_pre.replace("/", "-")) + '/'
    if not os.path.isdir(dir_save_vectors):
      os.makedirs(dir_save_vectors)

    if collection == 'deepmatcher' or collection == 'sparkly-all':
      url_save_vectorsA = dir_save_vectors + str(dataset)+ '_' + str(model_pre.replace("/", "-")) + '_ft_' + str(is_model_tuned) + '_vectors_TableA.csv'
      url_save_vectorsB = dir_save_vectors + str(dataset)+ '_' + str(model_pre.replace("/", "-")) + '_ft_' + str(is_model_tuned) + '_vectors_TableB.csv'
      #Converter para DF e incluir ID do registro na primeira coluna de cada linha
      vetores_dfA = pd.DataFrame(vetoresA)
      vetores_dfB = pd.DataFrame(vetoresB)
      vetores_dfA.insert(0,-1,dfIdsA.id,True)
      vetores_dfB.insert(0,-1,dfIdsB.id,True)
      #salvar arquivo de vetores em arquivos csv
      vetores_dfA.to_csv(url_save_vectorsA, header=False, index=False)
      vetores_dfB.to_csv(url_save_vectorsB, header=False, index=False)
      print('-> Vetores salvos em:')
      print('Vetores de TableA:',url_save_vectorsA)
      print('Vetores de TableB:',url_save_vectorsB)
    else:
      url_save_vectors = dir_save_vectors + 'vectors_' +str(collection)+ '_' + str(type_dataset) + '_' + str(model_pre.replace("/", "-")) + '_ft_' + str(is_model_tuned) + '.csv'
      #Converter para DF e incluir ID do registro na primeira coluna de cada linha
      vetores_df = pd.DataFrame(vetores)
      vetores_df.insert(0,-1,dfIds.id,True)
      #salvar arquivo de vetores em um csv
      vetores_df.to_csv(url_save_vectors, header=False, index=False)
      print('-> Vetores salvos em:',url_save_vectors)

#Verificar se os vetores já foram gerados previamente
else:
  if (collection == 'deepmatcher' or collection == 'sparkly-all') and ('vetoresA' in locals() or 'vetoresA' in globals()):
    print('-> Vetores já gerados previamente!')
  elif 'vetores' in locals() or 'vetores' in globals():
    print('-> Vetores já gerados previamente!')

if load_save_vectors:
  print('-> Carregando vetores do arquivo')
  dir_save_vectors = path_full + 'vectors/'+ str(model_pre.replace("/", "-")) + '/'
  if collection == 'deepmatcher' or collection == 'sparkly-all':

    try:
      url_save_vectorsA = dir_save_vectors + str(dataset)+ '_' + str(model_pre.replace("/", "-")) + '_ft_' + str(is_model_tuned) + '_vectors_TableA.csv'
      url_save_vectorsB = dir_save_vectors + str(dataset)+ '_' + str(model_pre.replace("/", "-")) + '_ft_' + str(is_model_tuned) + '_vectors_TableB.csv'

      vetoresA_com_id = genfromtxt(url_save_vectorsA, delimiter=',',dtype='float32')
      vetoresB_com_id = genfromtxt(url_save_vectorsB, delimiter=',',dtype='float32')
      # ids já coletados na leitura inicial do dataset em dfIdsA e dfIdsB
      vetoresA = np.delete(vetoresA_com_id, 0, 1) # apagar a coluna com ID
      vetoresB = np.delete(vetoresB_com_id, 0, 1) # apagar a coluna com ID
      print('VetoresA e VetoresB carregados:')
      print('VetoresA:',url_save_vectorsA)
      print('VetoresB:',url_save_vectorsB)
    except:
      print('Erro ao carregar vetores. Verifique se existem os arquivos para o dataset e modelo informado.')
  else:
    try:
      url_save_vectors = dir_save_vectors + 'vectors_' +str(collection)+ '_' + str(type_dataset) + '_' + str(model_pre.replace("/", "-")) + '_ft_' + str(is_model_tuned) + '.csv'
      vetores_com_id = genfromtxt(url_save_vectors, delimiter=',',dtype='float32')
      # ids já coletados na leitura inicial do dataset em dfIdsA e dfIdsB
      vetores = np.delete(vetores_com_id, 0, 1) # apagar a coluna com ID
      print('Vetores carregados com sucesso:')
      print('Vetores:',url_save_vectors)
    except:
       print('Erro ao carregar vetores. Verifique se existe o arquivo para o dataset e modelo informado.')

if perform_join:
  print('\n3. JUNÇÃO')
  log_result = []
  log_result.append('---- Informações da junção ----')

  if auto_threshold:
    match data_use:
      case 'train':
        dfDataUse = dfTrain
      case 'valid':
        dfDataUse = dfValid
      case 'test':
        dfDataUse = dfTest
      case 'train+valid':
        dfDataUse = pd.concat([dfTrain, dfValid], ignore_index=True)
      case _:
        raise ValueError(f"Invalid value for dataUse: {data_use}")
    print('-> Determinando melhor threshold')
    log_result.append('Auto-treshold ativado!')
    print('Junção em amostra: ',join_sample)
    start = time.time()
    if index == 'PairsInTestOnly':
      if collection == 'deepmatcher' or collection == 'sparkly-all':
        resultados,timeJoin = join_test_two_tables(dfDataUse,vetoresA,vetoresB,dfIdsA,dfIdsB,threshold,sample)
        threshold, best_f1 = find_best_threshold_two_tables(resultados,dfDataUse)
      else:
        resultados,timeJoin = join_test_one_table(dfDataUse,vetores,dfIds,threshold)
        threshold, best_f1 = find_best_threshold_auto_join_test(resultados,dfDataUse)
    else:
      if join_sample:
        log_result.append('- Tamanho da amostra para auto-threshold: ' + str(sample))
        if collection == 'deepmatcher' or collection == 'sparkly-all':
          resultados,timeIndex,timeJoin = join_tables_sample(vetoresA,vetoresB,dfIdsA,dfIdsB,index,threshold,sample,dfDataUse)
          threshold, best_f1 = find_best_threshold_two_tables(resultados,dfDataUse)
        else:
          print('Not implemented')
      else:
        if collection == 'deepmatcher' or collection == 'sparkly-all':
          resultados,timeIndex,timeJoin = join_tables(vetoresA,vetoresB,dfIdsA,dfIdsB,index,threshold)
          threshold, best_f1 = find_best_threshold_two_tables(resultados,dfDataUse)
        else:
          resultados,timeIndex,timeJoin = auto_join(vetores,dfIds,index,threshold)
          threshold, best_f1 = find_best_threshold_auto_join(resultados,dfIds)

    end = time.time()
    log_result.append('Threshold definido: ' + str(threshold) +' - Tempo para busca do threshold: %2.fms' %((end-start)*1000))
    #print('Threshold definido:',threshold,' - Tempo para busca do threshold: %2.fms' %((end-start)*1000))
    if join_sample:
      print('Tamanho da Amostra: ',sample)
  print('-> Iniciando operação de junção')
  log_result.append('Modelo: '+str(model_pre.replace("/", "-")))
  log_result.append('Modelo com fine-tuning: '+is_model_tuned)
  log_result.append('Tipo de índice: '+str(index))
  log_result.append('Threshold: '+str(threshold))

  # junção em duas tabelas
  if collection == 'deepmatcher' or collection == 'sparkly-all':

    if index == 'PairsInTestOnly':
      resultados,timeJoin = join_test_two_tables(dfTest,vetoresA,vetoresB,dfIdsA,dfIdsB,threshold)
    else:
      resultados,timeIndex,timeJoin = join_tables(vetoresA,vetoresB,dfIdsA,dfIdsB,index,threshold)
      log_result.append('Tempo de construção do índice: %2.fms' % (timeIndex*1000))

  # auto junção
  else:
    if index == 'PairsInTestOnly':
      resultados,timeJoin = join_test_one_table(dfTest,vetores,dfIds,threshold)
    else:
      resultados,timeIndex,timeJoin = auto_join(vetores,dfIds,index,threshold)
      log_result.append('Tempo de construção do índice: %2.fms' % (timeIndex*1000))


  log_result.append('Tempo da junção: %2.fms' % (timeJoin*1000))
  log_result.append('Quantidade de pares: '+str(len(resultados)))
  log_result = "\n".join(log_result)
  print(log_result)

  if show_metrics:

    if collection == 'deepmatcher' or collection == 'sparkly-all':
      print('-> Métricas do resultado da operação (Base rótulos Fixa)')
      log_metrics = get_metrics_sklearn_two_tables(resultados,dfTest)
    else:
      print('-> Métricas do resultado da auto-junção')
      if index == 'PairsInTestOnly':
        log_metrics = get_metrics_auto_join_test(resultados,dfTest)
      else:
        log_metrics = get_metrics_auto_join_full(resultados,dfIds)
    print(log_metrics)
    log_result += "\n-- Métricas ---\n"
    log_result += log_metrics
    f1score = float(re.search(r"F1-Score: (\d+\.\d+)", log_metrics).group(1))

  if save_log_metrics:
    print('-> Salvando log da junção e métricas em arquivo...')
    dir_path = path_full + 'result_joins/'+ str(model_pre.replace("/", "-")) + '/' + 'ft_' + str(is_model_tuned) +'/'
    if not os.path.isdir(dir_path):
      os.makedirs(dir_path)
    if join_sample:
      url_file_log = dir_path + str(index) + '_sample' + str(join_sample) + '-' + str(data_use) + '_' + str(sample) + '_ths' + str(threshold) + '_f1score' + str(f1score) + '_log.txt'
    else:
      url_file_log = dir_path + str(index) + '_sample' + str(join_sample) + '_ths' + str(threshold) + '_f1score' + f1score + '_log.txt'
    with open(url_file_log, 'w', encoding='utf-8') as f:
      f.write(log_result.encode('utf-8').decode('utf-8'))
    print('-> Log da junção salvo em',url_file_log)
    
  if save_result:
    print('-> Salvando resultado da junção em arquivo...')
    dir_path = path_full + 'result_joins/'+ str(model_pre.replace("/", "-")) + '/' + 'ft_' + str(is_model_tuned) +'/'
    if not os.path.isdir(dir_path):
      os.makedirs(dir_path)
    url_file_result = dir_path + str(index) + '_' + str(threshold) + '_result.csv'
    dfR = pd.DataFrame(resultados)
    dfR.to_csv(url_file_result, header=False, index=False)
    print('-> Resultado salvo em',url_file_result)
    print('-> Log da junção salvo no mesmo diretório')

endPip = time.time()
print('\n Tempo total da execução do pipeline % 2.fs '% (endPip-startPip) )
