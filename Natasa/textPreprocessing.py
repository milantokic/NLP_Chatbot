from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk

import numpy as np

#nltk.download('wordnet')

def text_preprocessing(dataset_question):
    question_list = [question for question in dataset_question]
    big_question_list = ' '.join(question_list)
    tokens = word_tokenize(big_question_list)
    words = [word.lower() for word in tokens if word.isalpha()]
    #stop_words = set(stopwords.words('english'))
    #words = [word for word in words if not in stop_words]
    return words

# Here, we need each question to remain a question --> basic
def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    #doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    return doc


# Here, we need each question to remain a question with stemming
s_stemmer = SnowballStemmer(language='english')
def preprocess_stem(text):
    text = text.lower()
    doc = word_tokenize(text)
    #doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    #doc = [word for s_stemmer.stem(word) in doc]

    docc = []
    for word in doc:
        stem_word = s_stemmer.stem(word)
        docc.append(stem_word)
    return docc

# Here, we need each document to remain a question with lemmatization
lemmatizer = WordNetLemmatizer()
def preprocess_lemma(text):
    text = text.lower()
    doc = word_tokenize(text)
    #doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    #doc = [word for word in doc if lemmatizer.lemmatize(word)]
    docc = []
    for word in doc:
        lemma_word = lemmatizer.lemmatize(word, pos="v")
        docc.append(lemma_word)
    return docc

def document_vector_mean(wv, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in wv.key_to_index]
    return np.mean(wv[doc], axis=0)

def document_vector_sum(wv, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in wv.key_to_index]
    return np.sum(wv[doc], axis=0)

# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(wv, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in wv.key_to_index for word in doc)

# Filter out documents
def filter_docs(corpus, texts, condition_on_doc):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)
    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]
    print("{} questions removed".format(number_of_docs - len(corpus)))
    return (corpus, texts)

def document_vector_weighted_sum(wv, doc, idf_words):
    doc = [word for word in doc if word in wv.key_to_index]

    list_of_idfs = []
    for word in doc:
        if word in idf_words:
            weighted_word_idf = idf_words[word] * wv[word]
            list_of_idfs.append(weighted_word_idf)
        else:
            weighted_word_idf = 0 * wv[word]
            list_of_idfs.append(weighted_word_idf)

    return np.sum(list_of_idfs, axis=0)

def document_vector_weighted_mean(wv, doc, idf_words):
    doc = [word for word in doc if word in wv.key_to_index]

    list_of_idfs = []
    for word in doc:
        if word in idf_words:
            weighted_word_idf = idf_words[word] * wv[word]
            list_of_idfs.append(weighted_word_idf)
        else:
            weighted_word_idf = 0 * wv[word]
            list_of_idfs.append(weighted_word_idf)

    return np.mean(list_of_idfs, axis=0)

def create_idf_list(tfidf):
    word_num = len(tfidf.vocabulary_)
    list_combine = {}
    for index in range(0,word_num):
        w_idf = tfidf.idf_[index]
        w = tfidf.get_feature_names()[index]
        list_combine[w] = w_idf
    return list_combine