#Import the libraries used for counting the tf
import numpy as np
import pandas as pd
import distances as dis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

#import dataset using pandas
dataset = pd.read_csv('insurance_qna_dataset.csv', sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
dataset_question = dataset_group_by_question.iloc[:, 0]

#Using the TfIdf Vectorizer to calculate the tf value for words in questions
tfidf = TfidfVectorizer()
count_tfidf = tfidf.fit_transform(dataset_question)

#Ask an insurance related question
new_question = [input("Please enter your insurance related question: ")]
count_tfidf_new_question = tfidf.transform(new_question)
print('\n')

number_of_question = len(dataset_question)
count_tfidf_new_question_todense = count_tfidf_new_question.toarray()
count_tfidf_todense = count_tfidf.toarray()
number_of_words = len(count_tfidf_todense[0])

the_top_similar = 20
'''
print(f"This are the {the_top_similar} most similar questions according to Manhattan distances: \n")
dis.manth_dist(number_of_question, count_tfidf_todense, count_tfidf_new_question_todense, the_top_similar, dataset_question)

print(f"This are the {the_top_similar} most similar question according to euclidean distances: \n")
dis.eucld_dist(number_of_question, count_tfidf_todense, count_tfidf_new_question_todense, the_top_similar, dataset_question)
'''

print(f"This are the most {the_top_similar} similar question according to cosine similarity: \n")
dis.cosin_dist_tf(number_of_question, count_tfidf_todense, count_tfidf_new_question_todense, the_top_similar, dataset_question)

'''
manhattan_dist = manhattan_distances(count_tfidf.toarray(), count_tfidf_new_question.toarray()).flatten()
print("This are the most similar questions according to manhattan distances: ")
print(dataset_question[manhattan_dist.argsort() [:the_top_similar]])

euclidean_dist = euclidean_distances(count_tfidf.toarray(), count_tfidf_new_question.toarray()).flatten()
print("This are the most similar questions according to euclidean distances: ")
print(dataset_question[euclidean_dist.argsort() [:the_top_similar]])
'''

cosine_similarity = cosine_similarity(count_tfidf.toarray(), count_tfidf_new_question.toarray()).flatten()
print("This are the most similar questions according to cosine similarity: ")
print(dataset_question.iloc[cosine_similarity.argsort() [-the_top_similar:][::-1]])







