import numpy as np
from numpy import dot
from numpy.linalg import norm

def manth_dist(number_of_question,count_vector, count_new_vector, top_similar, dataset_question):
    new_list = []
    for question in range(0, number_of_question):
        num_q = question
        calculate_md = 0
        test = [num_q]
        calculate_md = np.sum(np.absolute(count_vector[num_q, :] - count_new_vector[0, :]))
        test.append(calculate_md)
        new_list.append(test)
    new_list = sorted(new_list, key=lambda x: x[1])
    manhatt_dist = np.array(new_list)

    for result in range(0, top_similar):
        best_match = new_list[result][0]
        best_cs = new_list[-result][1]
        print(f'{dataset_question[best_match]} --> {best_cs}')

def eucld_dist(number_of_question,count_vector, count_new_vector, top_similar, dataset_question):
    new_list = []
    for question in range(0, number_of_question):
        num_q = question
        calculate_ed = 0
        test = [num_q]
        calculate_ed = np.sqrt(np.sum((count_vector[num_q, :] - count_new_vector[0, :]) ** 2))
        test.append(calculate_ed)
        new_list.append(test)
    new_list = sorted(new_list, key=lambda x: x[1])
    euclid_dist = np.array(new_list)

    for result in range(0, top_similar):
        best_match = new_list[result][0]
        best_cs = new_list[-result][1]
        print(f'{dataset_question[best_match]} --> {best_cs}')

def cosin_dist(number_of_question,count_vector, count_new_vector, top_similar, dataset_question):
    new_list = []
    for question in range(0, number_of_question):
        num_q = question
        calculate_cs = 0
        test = [num_q]
        calculate_cs = dot(count_vector[num_q, :], count_new_vector[:]) / (
                    norm(count_vector[num_q, :]) * norm(count_new_vector[:]))
        test.append(calculate_cs)
        new_list.append(test)
    new_list = sorted(new_list, key=lambda x: x[1])
    cosine_dist = np.array(new_list)

    for result in range(1, top_similar + 1):
        best_match = new_list[-result][0]
        best_cs = new_list[-result][1]
        print(dataset_question[best_match])


def cosin_dist_tf(number_of_question,count_vector, count_new_vector, top_similar, dataset_question):
    new_list = []
    for question in range(0, number_of_question):
        num_q = question
        calculate_cs = 0
        test = [num_q]
        calculate_cs = dot(count_vector[num_q, :], count_new_vector[0, :]) / (
                    norm(count_vector[num_q, :]) * norm(count_new_vector[0, :]))
        test.append(calculate_cs)
        new_list.append(test)
    new_list = sorted(new_list, key=lambda x: x[1])
    cosine_dist = np.array(new_list)

    for result in range(1, top_similar + 1):
        best_match = new_list[-result][0]
        best_cs = new_list[-result][1]
        print(f'{dataset_question[best_match]} --> {best_cs}')