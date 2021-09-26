#Import the libraries used for counting the tf
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import test_algorithmsForHighRecall as thr

#TO DO find how to trim the questions from dataset
def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

#import dataset using pandas
dataset = pd.read_csv('insurance_qna_dataset.csv', sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
dataset_question = dataset_group_by_question.iloc[:, 0]
number_of_question = len(dataset_question)


# Basic Tf CountVectorizer
tf = CountVectorizer()
count_tf = tf.fit_transform(dataset_question)

# SnawBall  Tf CountVectorizer
stemmer = SnowballStemmer(language='english')
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(text):
    return (stemmer.stem(w) for w in analyzer(text))

stem_tf = CountVectorizer(analyzer=stemmed_words)
count_stem_tf = stem_tf.fit_transform(dataset_question)

# LemmaTokenizer Tf CountVectorizer
lemmatizer = WordNetLemmatizer()
analyzer = CountVectorizer().build_analyzer()
def lemmatizer_words(text):
    return (lemmatizer.lemmatize(w) for w in analyzer(text))

lemma_tf = CountVectorizer(analyzer=lemmatizer_words)
count_lemma_tf = lemma_tf.fit_transform(dataset_question)

'''
#Ask an insurance related question
new_question = [input("Please enter your insurance related question: ")]
count_tf_new_question = tf.transform(new_question)
print('\n')
'''

test_questions = {'0': 'What Is The Best Life Insurance To Buy? ',
                  '1': 'What Is The Comprehensive Life Insurance To Buy? ',
                  '2': 'What Is The Comprehensive Life Insurance To Purchase? ',
                  '3': 'What Is The Most Outstanding Guarantee To Purchase? ',

                  '4': 'How Much Does A Whole Life Insurance Policy Typically Cost? ',
                  '5': 'How Much Does A Whole Life Insurance Policy Usually Cost? ',
                  '6': 'What Is The Typically Price Of Whole Life Insurance Policy? ',
                  '7': 'At What Price Could Life Insurance Be Bought? ',

                  '8': 'Who Should Get Critical Illness Insurance? ',
                  '9': 'Who Should Buy Critical Illness Insurance? ',
                  '10': 'Critical Illness Insurance Should Be Got By Who? ',
                  '11': 'Which People Should Buy Censorious Illness Insurance? ',

                  '12': 'Where To Buy Good Life Insurance? ',
                  '13': 'Where To Get Quality Life Insurance? ',
                  '14': 'Where Is The Best Place To Buy Quality Life Insurance? ',
                  '15': 'Which Is The Best Insurance Company to Get Satisfying Life Insurance? ',

                  '16': 'How Can I Find Who My Car Insurance Is With? ',
                  '17': 'How To Find Who My Car Insurance Is With? ',
                  '18': 'How To Find Out Which Is My Vehicle Insurance Company? ',
                  '19': 'At Which Company My Vehicle Is Insured? ',

                  '20': 'What Does Home Insurance Cost? ',
                  '21': 'What Does Property Insurance Cost? ',
                  '22': 'What Is The Price Of Home Insurance? ',
                  '23': 'What Does Insuring My House Cost? ',

                  '24': 'What Is The Purpose Of Life Insurance? ',
                  '25': 'What Is The Motive Of Life Insurance? ',
                  '26': 'What Is The Idea Of Life Indemnity? ',
                  '27': 'What Is The Idea Of Buying Life Indemnity? ',

                  '28': 'Are New Cars Cheaper To Insure Than Older Cars? ',
                  '29': 'Are New Vehicles Cheaper To Insure Than Older? ',
                  '30': 'Is It Cheaper To Insure, Old Or New Cars? ',
                  '31': 'Is It More Expensive To Insure New Vehicles Or Old? ',

                  '32': 'What Is The Best Disability Insurance Company? ',
                  '33': 'What Is The Best Invalidity Insurance Company? ',
                  '34': 'What Is The Recommended Invalidity Insurance Company? ',
                  '35': 'At What Company Should I Buy Disability Insurance? ',

                  '36': 'How Much Money Does A Life Insurance Salesman Make? ',
                  '37': 'How Much Cash Does A Life Protections Sales Representative Make? ',
                  '38': 'How Much Cash Does A Life Assurances Deals Agent Make? ',
                  '39': 'What Is The Amount Of Cash That A Life Protections Sales Make? '
                  }

num_test_scenarios = len(test_questions) / 4
num_tests = len(test_questions)
the_top_similar = 30


thr.t_algorithm(test_questions, tf, count_tf, the_top_similar)
thr.t_algorithm(test_questions, lemma_tf, count_lemma_tf, the_top_similar)
thr.t_algorithm(test_questions, stem_tf, count_stem_tf, the_top_similar)


'''
count_tf_new_question_todense = count_tf_new_question.toarray()
count_tf_todense = count_tf.toarray()

#My created loops for calculating the Manhattan, Euclidean and Cosine distances
print(f"This are the {the_top_similar} most similar questions according to Manhattan distances: \n")
dis.manth_dist(number_of_question, count_tf_todense, count_tf_new_question_todense, the_top_similar, dataset_question)

print(f"This are the {the_top_similar} most similar question according to euclidean distances: \n")
dis.eucld_dist(number_of_question, count_tf_todense, count_tf_new_question_todense, the_top_similar, dataset_question)

print(f"This are the most {the_top_similar} similar question according to cosine similarity: \n")
dis.cosin_dist_tf(number_of_question, count_tf_todense, count_tf_new_question_todense, the_top_similar, dataset_question)
'''

'''
manhattan_dist = manhattan_distances(count_tf.toarray(), count_tf_new_question.toarray()).flatten()
print("This are the most similar questions according to manhattan distances: ")
print(dataset_question[manhattan_dist.argsort() [:the_top_similar]])

euclidean_dist = euclidean_distances(count_tf.toarray(), count_tf_new_question.toarray()).flatten()
print("This are the most similar questions according to euclidean distances: ")
print(dataset_question[euclidean_dist.argsort() [:the_top_similar]])

cosine_similarity = cosine_similarity(count_tf.toarray(), count_tf_new_question.toarray()).flatten()
print("This are the most similar questions according to cosine similarity: ")
print(dataset_question.iloc[cosine_similarity.argsort() [-the_top_similar:][::-1]])
'''







