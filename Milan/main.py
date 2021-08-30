import numpy as np
import pandas as pd
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import gensim.downloader

nltk.download('wordnet')
nltk.download('punkt')

def read_data(path):
    cr = pd.read_csv(path, sep='\t')
    return cr


lem = WordNetLemmatizer()
wv = gensim.downloader.load('word2vec-google-news-300')


def lemmatize_sent(sent):
    lem_sent = []
    for word in word_tokenize(sent):
        if word.isalnum():
            lem_sent.append(lem.lemmatize(word))
    return " ".join(lem_sent)


def sent_w2v(sent):
    vector_list = []
    for token in word_tokenize(lemmatize_sent(sent)):
        try:
            vector_list.append(wv[token])
        except:
            pass
    return vector_list


def sum_vectors(vector_list):
    return np.sum(vector_list, axis=0)


corpus = read_data('insurance_qna_dataset.csv').iloc[:, 1:]
corpus = corpus.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
corpus_q = corpus['Question']

all_vec = []
print("Input question: ")
new_q = input()

all_sim = [cosine_similarity([sum_vectors(sent_w2v(new_q))] , [sum_vectors(sent_w2v(corpus_q[i]))]) for i in range(corpus_q.shape[0])]

top_10 = np.argsort(all_sim, axis=None)[-10:][::-1]

