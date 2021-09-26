import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances

import textPreprocessing as tp

### import dataset using pandas ###
dataset = pd.read_csv("C:\\Users\\Natasa\\Documents\\Machine Learning\\insurance_qna_dataset.csv", sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
dataset_question = dataset_group_by_question.iloc[:, 0]
question_list = [question for question in dataset_question]

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

def t_algorithm(test_questions, tf, count_tf, the_top_similar):
    list_of_grades_cosine = []
    list_of_grades_manhattan = []
    list_of_grades_euclidean = []
    index = 0
    for key, value in test_questions.items():
        algorithm_grade = 0
        t_quest = value
        if index==0 or index%4==0:
            t_value = value

        count_tf_new = tf.transform([t_quest])
        count_tf_todense_new = count_tf_new.toarray()
        count_tf_todense = count_tf.toarray()

        #############################################
        #########Calculatin Cosine Similarity########
        #############################################

        cosine_sim = cosine_similarity(count_tf_todense, count_tf_todense_new).flatten()
        result_cosine = dataset_question.iloc[cosine_sim.argsort()[-the_top_similar:][::-1]]
        is_in_cosine = result_cosine.isin([t_value])

        index_grade_cosine = 0
        added_to_list_cosine = 'No'
        for grade in is_in_cosine:
            index_grade_cosine += 1
            if grade == True:
                added_to_list_cosine = 'Yes'
                list_of_grades_cosine.append(index_grade_cosine)


        if added_to_list_cosine != 'Yes':
            fee = the_top_similar * 2
            list_of_grades_cosine.append(fee)

        #############################################
        #########Calculatin Manhattan Distance#######
        #############################################

        manhattan_dist = manhattan_distances(count_tf_todense, count_tf_todense_new).flatten()
        result_manhattan = dataset_question[manhattan_dist.argsort() [:the_top_similar]]
        is_in_manhattan = result_manhattan.isin([t_value])

        index_grade_manhattan = 0
        added_to_list_manhattan = 'No'
        for grade in is_in_manhattan:
            index_grade_manhattan += 1
            if grade == True:
                added_to_list_manhattan = 'Yes'
                list_of_grades_manhattan.append(index_grade_manhattan)

        if added_to_list_manhattan != 'Yes':
            fee = the_top_similar * 2
            list_of_grades_manhattan.append(fee)

            #############################################
            #########Calculatin Euclidean Distance#######
            #############################################

            euclidean_dist = euclidean_distances(count_tf_todense, count_tf_todense_new).flatten()
            result_euclidean = dataset_question[euclidean_dist.argsort()[:the_top_similar]]
            is_in_euclidean = result_euclidean.isin([t_value])

            index_grade_euclidean = 0
            added_to_list_euclidean = 'No'
            for grade in is_in_euclidean:
                index_grade_euclidean += 1
                if grade == True:
                    added_to_list_euclidean = 'Yes'
                    list_of_grades_euclidean.append(index_grade_euclidean)

            if added_to_list_euclidean != 'Yes':
                fee = the_top_similar * 2
                list_of_grades_euclidean.append(fee)

        index += 1
        algorithm_grade_cosine = sum(list_of_grades_cosine)
        algorithm_grade_manhattan = sum(list_of_grades_manhattan)
        algorithm_grade_euclidean = sum(list_of_grades_euclidean)

    print(f'Score with Cosine Similarity is: {algorithm_grade_cosine}')
    print(f'Score with Manhattan Distance is: {algorithm_grade_manhattan}')
    print(f'Score with Euclidean Distance is: {algorithm_grade_euclidean}')



def t_algorithm_gensim(operation, preprocessing, test_questions, wv, X, the_top_similar, idf_words):
    #default operation parameter is sum
    #default preprocessing parameter is basic
    list_of_grades_cosine = []
    list_of_grades_manhattan = []
    list_of_grades_euclidean = []
    index = 0
    for key, value in test_questions.items():
        algorithm_grade = 0
        t_quest = value
        if index == 0 or index % 4 == 0:
            t_value = value

        if preprocessing == 'stem':
            new_question_basic = tp.preprocess_stem(t_quest)
        elif preprocessing == 'lemma':
            new_question_basic = tp.preprocess_lemma(t_quest)
        else:
            new_question_basic = tp.preprocess(t_quest)

        if operation == 'mean':
            new_question_basic = tp.document_vector_mean(wv, new_question_basic).reshape(1, -1)
        elif operation == 'weightedSum':
            new_question_basic = tp.document_vector_weighted_sum(wv, new_question_basic, idf_words).reshape(1, -1)
        elif operation == 'weightedMean':
            new_question_basic = tp.document_vector_weighted_mean(wv, new_question_basic, idf_words).reshape(1, -1)
        else:
            new_question_basic = tp.document_vector_sum(wv, new_question_basic).reshape(1, -1)

        #############################################
        #########Calculatin Cosine Similarity########
        #############################################

        cosine_sim = cosine_similarity(X, new_question_basic).flatten()
        result_cosine = dataset_question.iloc[cosine_sim.argsort()[-the_top_similar:][::-1]]
        is_in_cosine = result_cosine.isin([t_value])

        index_grade_cosine = 0
        added_to_list_cosine = 'No'
        for grade in is_in_cosine:
            index_grade_cosine += 1
            if grade == True:
                added_to_list_cosine = 'Yes'
                list_of_grades_cosine.append(index_grade_cosine)

        if added_to_list_cosine != 'Yes':
            fee = the_top_similar * 2
            list_of_grades_cosine.append(fee)

        #############################################
        #########Calculatin Manhattan Distance#######
        #############################################

        manhattan_dist = manhattan_distances(X, new_question_basic).flatten()
        result_manhattan = dataset_question[manhattan_dist.argsort()[:the_top_similar]]
        is_in_manhattan = result_manhattan.isin([t_value])

        index_grade_manhattan = 0
        added_to_list_manhattan = 'No'
        for grade in is_in_manhattan:
            index_grade_manhattan += 1
            if grade == True:
                added_to_list_manhattan = 'Yes'
                list_of_grades_manhattan.append(index_grade_manhattan)

        if added_to_list_manhattan != 'Yes':
            fee = the_top_similar * 2
            list_of_grades_manhattan.append(fee)

            #############################################
            #########Calculatin Euclidean Distance#######
            #############################################

            euclidean_dist = euclidean_distances(X, new_question_basic).flatten()
            result_euclidean = dataset_question[euclidean_dist.argsort()[:the_top_similar]]
            is_in_euclidean = result_euclidean.isin([t_value])

            index_grade_euclidean = 0
            added_to_list_euclidean = 'No'
            for grade in is_in_euclidean:
                index_grade_euclidean += 1
                if grade == True:
                    added_to_list_euclidean = 'Yes'
                    list_of_grades_euclidean.append(index_grade_euclidean)

            if added_to_list_euclidean != 'Yes':
                fee = the_top_similar * 2
                list_of_grades_euclidean.append(fee)

        index += 1
        algorithm_grade_cosine = sum(list_of_grades_cosine)
        algorithm_grade_manhattan = sum(list_of_grades_manhattan)
        algorithm_grade_euclidean = sum(list_of_grades_euclidean)

    print(f'Score with Cosine Similarity is : {algorithm_grade_cosine} with {preprocessing} and {operation}')
    print(f'Score with Manhattan Distance is: {algorithm_grade_manhattan} with {preprocessing} and {operation}')
    print(f'Score with Euclidean Distance is: {algorithm_grade_euclidean} with {preprocessing} and {operation}')