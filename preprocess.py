#!/usr/bin/python
import gensim
import numpy as np
import random
import json
import utils
from node2vec_files import node2vec_functions

# ***************** In this script, all the preprocess steps in order to create the necessary files for
# ***************** running co-trained ensemble are done
# ***************** please run each section separately from top to down
# ***************** when running each section, comment the remaining sections


data_set = 'synthetic'
dataset = 'synthetic.txt'

# # # section 1
# # # *******************************************************************
# # #***************** shuffle the data, and assign an ID for each message.
# # # *******************************************************************
msg_list = []
tweet_counter = 0
with open('./data/' + data_set + '/files/' + dataset) as json_file:
    data = json.load(json_file)
    for t in data['tweet']:
        if t['target_name'] != 'None' and t['target_name'] != t['user_name'] and t['text'] != '':
            msg_list.append(t)

shuffle_msg_arr = np.array(msg_list)
random.shuffle(shuffle_msg_arr)

for i, dic in enumerate(shuffle_msg_arr):
    dic['ID'] = i

f_1 = open('./data/' + data_set + '/files/'+'/all.txt','w')
data = {}
data['tweet'] = list(shuffle_msg_arr)
json.dump(data,f_1)
f_1.close()

# # # section 2
# *******************************************************************
# ***************** divide  dataset to train and validation sets
# *******************************************************************
with open('./data/' + data_set + '/files/'+'/all.txt','r') as json_file:
    data = json.load(json_file)
    num_conversations = len(data['tweet'])
    num_training = (num_conversations/100)* 80
    num_val = num_conversations - num_training

    f_t = open('./data/' + data_set + '/files/'+'/training.txt','w')
    f_v = open('./data/' + data_set + '/files/'+'/validation.txt','w')
    counter = 0
    data_train = {}
    data_val = {}
    data_train['tweet'] = []
    data_val['tweet'] = []
    for line in data['tweet']:
        if counter <= num_training:
            data_train['tweet'].append(line)
        else:
            data_val['tweet'].append(line)

        counter += 1
    json.dump(data_train,f_t)
    json.dump(data_val,f_v)

# # # section 3
# *******************************************************************
# ***************** create Lower bound (LB) and Upper bound (UB) for each message
# ***************** load positive and negative list of words and create a LB and UB
# ***************** create the edges file; we need this file for computing node2vec
# ***************** these files will be saved in 'saved_models' directory
# **************** please read the paper for more information
# *******************************************************************
seed_words = set([])
f = open('./data/'+data_set+'/files/badwords.txt', 'r')
for s_w in f:
    seed_words.add(s_w.strip())

pos_words = set([])
f = open('./data/'+data_set+'/files/positive-words.txt', 'r')
for p_w in f:
    pos_words.add(p_w.strip())
f.close()
f_nodes = open('./saved_models/'+ data_set +'/nodes.edgelist', 'w')
LB = []
UB = []
bounds_f = open('./saved_models/'+ data_set +'/bounds.txt', 'w')
cnt = 0
bounds_dic = {}
bounds_dic['bounds'] = []
counter = 0
with open('./data/' + data_set + '/files/'+'/all.txt','r') as json_file:
    data = json.load(json_file)
    for t in data['tweet']:
        counter += 1
        dic = {}
        word_lists = t['text'].strip().split()
        word_lists = [word.lower() for word in word_lists]
        dic['ID'] = t['ID']
        if len(set(word_lists).intersection(seed_words)) == 0:
            dic['LB'] = 0
        else:
            dic['LB'] = len(set(word_lists).intersection(seed_words)) / len(word_lists)

        if len(set(word_lists).intersection(pos_words)) == 0:
            dic['UB'] = 1
        else:
            dic['UB'] = 1 - (len(set(word_lists).intersection(pos_words)) / len(word_lists))

        bounds_dic['bounds'].append(dic)


        f_nodes.write(str(t['user_name']) + '    ' + str(t['target_name']))
        f_nodes.write('\n')

    json.dump(bounds_dic,bounds_f)

f_nodes.close()
bounds_f.close()

# # # section 4
# # *******************************************************************
# ***************** train Word2vec model and save the model
# ***************** in 'saved_models' directory
# # *******************************************************************
sentences = []
with open('./data/' + data_set + '/files/'+'/all.txt','r') as json_file:
    data = json.load(json_file)
    for t in data['tweet']:
        word_lists = t['text'].strip().split()
        sentences.append(word_lists)

model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1)
model.save('./saved_models/'+ data_set +'/word2vec_model')

# # # section 5
# # *******************************************************************
# ***************** train doc2vec model and save the model
# ***************** in 'saved_models' directory
# # *******************************************************************
sentences = []
with open('./data/' + data_set + '/files/'+'/all.txt','r') as json_file:
    data = json.load(json_file)
    for t in data['tweet']:
        word_lists = t['text'].strip().split()
        sentences.append((t['ID'],word_lists))


unsup_reviews = utils.labelize_mesages(sentences, 'UNSUP')
model_dm = gensim.models.Doc2Vec(min_count=1, window=4, size=100, alpha=0.025, min_alpha=0.025)
model_dm.build_vocab(unsup_reviews)

for epoch in range(10):
    print(epoch)
    model_dm.train(unsup_reviews)
    model_dm.alpha -= 0.002  # decrease the learning rate
    model_dm.min_alpha = model_dm.alpha  # fix the learning rate, no decay

# store the model
model_dm.save('./saved_models/'+ data_set +'/doc2vec_model')

# # # section 6
# # *******************************************************************
# ***************** train Node2vec model and save the model
# ***************** in 'saved_models' directory
# ***************** we downloaded the necessary files from SNAP repository
# ***************** https://snap.stanford.edu/node2vec/
# # *******************************************************************
input_add = './saved_models/'+ data_set +'/nodes.edgelist'
output_add ='./saved_models/'+ data_set +'/output_nodes.emb'
args = node2vec_functions.parse_args(input_add, output_add)
nx_G = node2vec_functions.read_graph(args)
G = node2vec_functions.node2vec.Graph(nx_G, args.directed, args.p, args.q)
G.preprocess_transition_probs()
walks = G.simulate_walks(args.num_walks, args.walk_length)
node2vec_functions.learn_embeddings(walks, args)

