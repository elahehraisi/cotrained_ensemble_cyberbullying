#!/usr/bin/python
import numpy as np
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse import csr_matrix
import operator
import json
import os


# this script is the implementation of participant vocabulary consistency
# for more information please read the paper "Cyberbullying Detection with Weakly Supervised Machine Learning"
# http://people.cs.vt.edu/~bhuang/papers/raisi-asonam17.pdf

# directory in which the results will be saved
save_dir = './data/synthetic/saved_data/PVC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# cleaned data set. All of the preprocessing steps have been done in cleaned data. The data is saved in json format
dataset = 'synthetic.txt'


# this function computes the bullying score of interactions
def get_score(b,v,w,messages,lam,has_fouls):
    b_v_sum = np.add(b,v)
    b_v_sum_squar = np.power(b_v_sum,2)
    w_sum = messages.sum(axis=1)
    w_sum = np.add(w_sum.T,has_fouls)
    p1 = np.multiply(w_sum,b_v_sum_squar)
    
    p2 = np.add(messages.dot(np.power(w,2)) , np.power(has_fouls,2))
    
    p3 = -2 * np.multiply(b_v_sum,np.add(messages.dot(w),has_fouls))
    
    bully_score = p1 + p2 + p3

    unique_b = np.unique(b)
    unique_v = np.unique(v)

    reg = np.dot(unique_b,np.transpose(unique_b)) + np.dot(unique_v,np.transpose(unique_v)) + np.dot(w,np.transpose(w))
    score = .5 * np.sum(bully_score)+ 0.5 * lam * reg
    return score


def main():
    # read seed words file to fouls list.
    fouls = []
    with open('./data/synthetic/files/badwords.txt') as f:
        for key in f:
            if key:
                fouls.append(key.strip())



    # ####################################################################
    # ######**************** Load the cleaned data, create the word list
    # ####################################################################
    bi_list = []
    bully = []
    victim = []
    words_list = []
    has_fouls = []
    message_text = []
    with open('./data/synthetic/files/' +  dataset) as json_file:
        data = json.load(json_file)
        for t in data['tweet']:
            wd = []
            value = t['text']
            for w in value.split():
                if w.strip().lower() not in fouls:
                    wd.append(w.strip())
            c = 0
            for fl in fouls:
                if fl in t['text'].lower():
                    c += 1

            words = list(set(wd))
            if t['target_name'] != 'None' and t['target_name'] != t['user_name'] and words != []:
                bully.append(t['user_name'])
                victim.append(t['target_name'])
                words_list.extend(words)
                message_text.append(t['text'])
                has_fouls.append(c)

    bully = np.array(bully)
    victim = np.array(victim)

    # ####################################################################
    # ######**************** initialize bully, victim and matrix of messages
    # ####################################################################

    # load bigrams
    with open('./data/synthetic/files/bigrams.txt') as f:
        for key in f:
            bi_list.append(key.strip())

    # combine words and bigrams
    words = list(set(words_list)) + bi_list

    num_message = len(bully)
    num_words = len(words)
    final_users = np.concatenate((bully, victim), axis=0)

    print('number of messages:' + str(num_message))
    print('number of words:' + str(num_words))
    print('total number of users:' + str(len(list(set(final_users)))))

    # initialize bully, victim, and word scores
    b0 = np.ones(len(bully)) * [0.1]
    v0 = np.ones(len(victim)) * [0.1]
    w0 = np.ones((num_words)) * [0.1]
    lamda = 8

    # create BoW matrix (words are unigrams and bigrams)
    df = pd.DataFrame(message_text, columns=['text'])
    countvec = CountVectorizer(vocabulary=words, ngram_range=(1, 2), binary=True, lowercase=False)
    messages = countvec.fit_transform(df.text)


    b = b0
    v = v0
    w = w0

    # prepare for updating bully score (b) according to equation 4
    rows = []
    cols = []
    unique_b = np.unique(bully)
    bully_series = pd.Series(bully)
    bully_groups = bully_series.groupby(bully_series.values)
    c = 0
    for name, group in bully_groups:
        itemindex = list(((group).index).astype(np.int))
        for j in itemindex:
            rows.append(c)
            cols.append(j)
        c += 1

    data = np.ones(len(rows))
    C_bully = csr_matrix((data, (rows, cols)), shape=(len(unique_b), num_message))

    L = messages.sum(axis=1)
    L = np.squeeze(np.asarray(L))
    L = np.add(L.T, has_fouls)
    L = L.T
    d = C_bully.dot(L) + lamda
    denom_bully = 1. / d


    # prepare for updating victim score (v) according to equation 5
    rows = []
    cols = []
    unique_v = np.unique(victim)
    victim_series = pd.Series(victim)
    victim_groups = victim_series.groupby(victim_series.values)
    c = 0
    for name, group in victim_groups:
        itemindex = list(((group).index).astype(np.int))
        for j in itemindex:
            rows.append(c)
            cols.append(j)
        c += 1

    data = np.ones(len(rows))
    C_victim = csr_matrix((data, (rows, cols)), shape=(len(unique_v), num_message))

    # prepare for updating word score (w) according to equation 6
    d1 = C_victim.dot(L) + lamda
    denom_victim = 1. / d1

    d2 = messages.sum(axis=0) + lamda
    di2 = (1. / d2)
    denom_word = np.squeeze(np.asarray(di2))
    messages_t = messages.T

    tb, ind_b = np.unique(bully, return_inverse=True)
    tv, ind_v = np.unique(victim, return_inverse=True)

    # ####################################################################
    # ######**************** start training PVC using Alternating Least Square to get the score of users and words
    # ####################################################################
    max_itr = 500
    print('start ALS...')
    for j in range(0, max_itr):
        # update bully vector (b): Equation 4
        diff = np.subtract(np.add(messages.dot(w), has_fouls), np.multiply(L.T, v))
        diff = np.squeeze(np.asarray(diff))
        b_grad = C_bully.dot(diff)
        b_new = np.multiply(b_grad, denom_bully)
        b_new = np.squeeze(np.asarray(b_new))
        b = [b_new[p] for p in ind_b]

        # update victim score (v): Equation 5
        diff1 = np.subtract(np.add(messages.dot(w), has_fouls), np.multiply(L.T, b))
        v_grad = C_victim.dot(diff1)
        v_new = np.multiply(v_grad, denom_victim)
        v1 = np.squeeze(np.asarray(v_new.T))
        v = [v1[p] for p in ind_v]

        #update words score (w): Equation 6
        b_v_sum = (np.add(b, v))
        Z2 = (messages_t.dot(b_v_sum.T).T)
        w = np.multiply(Z2, denom_word)

        score = get_score(b, v, w, messages, lamda, has_fouls)
        print(score)

    print('ALS done')


    # ####################################################################
    # ######**************** sort the words based on the scores (from high to low), and save them into results.txt
    # ####################################################################
    results = open(save_dir + 'results.txt', 'w')
    ind = (np.argsort(w)[::-1])
    detected_words = (itemgetter(*ind)(words))
    results.write(", ".join(detected_words))
    results.write('\n')
    results.write(', '.join(map(str, itemgetter(*ind)(w))))
    results.write('\n')


    # ####################################################################
    # ######**************** start computing the average bullying score for user pair,
    # ######**************** and sort them from high to low, and save them into PVC_bullying_interaction.txt
    # ####################################################################
    b_v_sum = (np.add(b, v))
    word_score = np.add(messages.dot(w), has_fouls)
    denum = np.array((np.sum(messages, axis=1)))

    denum = np.squeeze(denum) + has_fouls

    norm_word_score = np.true_divide(word_score, denum.T)

    interaction_score = b_v_sum + norm_word_score

    dic_bully = {}
    dic_victim = {}
    bully_list = list(bully)
    victim_list = list(victim)

    for i, b_l in enumerate(bully_list):
        dic_bully.setdefault(b_l, []).append(i)

    for i, v_l in enumerate(victim_list):
        dic_victim.setdefault(v_l, []).append(i)

    selected = set()
    dic_u = {}
    for i in range(len(bully)):
        b_u = bully[i]
        v_u = victim[i]
        if (b_u, v_u) not in selected and (v_u, b_u) not in selected:
            if dic_bully.get(b_u) is None:
                a1 = []
            else:
                a1 = dic_bully.get(b_u)
            if dic_victim.get(v_u) is None:
                a2 = []
            else:
                a2 = dic_victim.get(v_u)
            shared_1 = np.intersect1d(a1, a2)
            if dic_victim.get(b_u) is None:
                a1 = []
            else:
                a1 = dic_victim.get(b_u)
            if dic_bully.get(v_u) is None:
                a2 = []
            else:
                a2 = dic_bully.get(v_u)

            shared_2 = np.intersect1d(a1, a2)

            shared = np.union1d(shared_1, shared_2)
            shared = shared.astype(int)
            # get the conversation with at least one interaction
            if len(shared) >= 1:
                avr = 0
                for j in shared:
                    avr += interaction_score[j]
                dic_u[str(b_u) + ',' + str(v_u)] = np.true_divide(avr, len(shared))

            selected.add((b_u, v_u))
            selected.add((v_u, b_u))

    sorted_dic = sorted(dic_u.items(), key=operator.itemgetter(1), reverse=True)
    f = open(save_dir + '/PVC_bullying_interaction.txt', 'w')
    for key in sorted_dic:
        f.write(str(key[0]) + ',' + str(key[1]))
        f.write('\n')

    print('done')


if __name__ == "__main__":
    main()




