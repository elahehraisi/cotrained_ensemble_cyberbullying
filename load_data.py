import os
import torch
import numpy as np
import gensim
import json

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.dictionary.add_word('ZERO_TOKEN')

        self.messages_list = []
        self.messages_list_remove_sensitives = []
        self.max_len = 0

        self.train = self.tokenize(os.path.join(path+'/files/', 'training.txt'))
        self.valid = self.tokenize(os.path.join(path+'/files/', 'validation.txt'))

    def get_message_ix_list(self,path):
        self.train_message_ix = self.get_message_ix(os.path.join(path+'/files/', 'training.txt'))
        self.valid_message_ix = self.get_message_ix(os.path.join(path+'/files/', 'validation.txt'))
        self.all_message_ix = self.get_message_ix(os.path.join(path+'/files/', 'all.txt'))

    def get_list_message(self,path):
        path = os.path.join(path, 'all.txt')
        with open(path, 'r') as f:
            for line in f:
                self.messages_list.append((line.strip()))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            data = json.load(f)
            for t in data['tweet']:
                self.messages_list.append((t['text'].strip().strip()))
                words = t['text'].strip().split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)


        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            data = json.load(f)
            for t in data['tweet']:
                line = t['text'].strip()
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def get_message_ix(self,path):
        assert os.path.exists(path)

        message_ix = []
        with open(path,'r') as json_file:
            data = json.load(json_file)
            for t in data['tweet']:
                message_words = t['text'].strip().split()
                m_ix = []
                for w in message_words:
                    if w in self.dictionary.word2idx:
                        m_ix.append(self.dictionary.word2idx[w])
                    else:
                        print('err: word ' + w + ' does not exist in dictionary')
                        self.dictionary.add_word(w)
                        m_ix.append(self.dictionary.word2idx[w])

                if len(m_ix) > self.max_len:
                    self.max_len = len(m_ix)
                message_ix.append(m_ix)

            # ## zero pad the messages (if there are some very long messages in the data, zero padding should be done for each batch)
            for ix in message_ix:
                # print(ix)
                if len(ix) < self.max_len:
                    # append_zero = []
                    index = 0
                    for i in range(self.max_len - len(ix)):
                        ix.insert(index, 0)
                        index += 1



        return message_ix

    def get_pretrained_matrix_msg(self,dataset):
        assert os.path.exists('./saved_models/'+dataset+'/word2vec_model')
        model_dm = gensim.models.Word2Vec.load('./saved_models/'+dataset+'/word2vec_model')

        num_words = len(self.dictionary.word2idx)
        emb_list = []
        for i in range(num_words):
            if self.dictionary.idx2word[i] in model_dm.wv:
                emb_list.append(model_dm.wv[self.dictionary.idx2word[i]])
            else:
                emb_list.append(np.random.uniform(-0.1,0.1,100))

        return emb_list


class Pretrained(object):
    def __init__(self):
        self.usr_train = []
        self.usr_val = []
        self.usr_test = []
        self.usr_all = []
        self.doc2vec_train = []
        self.doc2vec_val = []
        self.doc2vec_all = []
        self.LB = []
        self.UB = []
        self.LB_train = []
        self.LB_val = []
        self.UB_train = []
        self.UB_val = []
        self.train_nodes_ix = []
        self.val_nodes_ix = []
        self.all_nodes_ix = []
        self.num_users_all = 0
        self.user_dictionary = Dictionary()
        self.user_b_v = {}
        self.doc2vec_all_removed_sensitives = []
        self.doc2vec_train_removed_sensitives = []
        self.doc2vec_val_removed_sensitives = []

    def node_tokenize(self,dataset):
        assert os.path.exists('./saved_models/'+ dataset+'/nodes.edgelist')
        f = open('./saved_models/'+ dataset+'/nodes.edgelist', 'r')
        nodes = []
        users = set()
        for line in f:
            s, r = line.split()
            self.user_dictionary.add_word(s.strip())
            self.user_dictionary.add_word(r.strip())

    def get_user_ix(self,num_training,dataset):
        assert os.path.exists('./saved_models/'+ dataset+'/nodes.edgelist')

        lines = []
        f = open('./saved_models/' + dataset + '/nodes.edgelist', 'r')
        for l in f:
            lines.append(l.strip())
        with open('./data/' + dataset + '/files/' + '/all.txt', 'r') as json_file:
            data = json.load(json_file)
            ID_counter = 0
            for t in data['tweet']:
                s, r = lines[ID_counter].split()
                assert str(t['user_name']) == str(s)
                assert str(t['target_name']) == str(r)
                ID_counter += 1


        f = open('./saved_models/'+ dataset+'/nodes.edgelist', 'r')
        nodes = []
        users = set()
        for line in f:
            s, r = line.split()
            s = s.strip()
            r = r.strip()
            if s in self.user_dictionary.word2idx.keys() and r in self.user_dictionary.word2idx.keys():
                nodes.append([self.user_dictionary.word2idx[s], self.user_dictionary.word2idx[r]])
            else:
                print('Err: user node does not exist in the dictionary?!!!')

            if not s in users:
                users.add(s)

            if not r in users:
                users.add(r)

        self.train_nodes_ix = nodes[:num_training]
        self.val_nodes_ix = nodes[num_training:]
        self.all_nodes_ix = nodes
        self.num_users_all = len(users)

    def get_pretrained_matrix_usr(self, dataset):
        assert os.path.exists('./saved_models/'+dataset+'/output_nodes.emb')
        nodes = {}
        f = open('./saved_models/'+dataset+'/output_nodes.emb', 'r')
        counter = 0
        for linen in f:
            if counter != 0:
                line_splt = linen.split()
                nodes[line_splt[0]] = list(line_splt[1:])

            counter += 1

        emb_list = []
        num_user = len(self.user_dictionary.idx2word)
        for i in range(num_user):
            emb_list.append(nodes[self.user_dictionary.idx2word[i]])

        return emb_list

    def get_pretrained_user(self,num_training, dataset):
        assert os.path.exists('./saved_models/'+dataset+'/output_nodes.emb')
        nodes = {}
        f = open('./saved_models/'+dataset+'/output_nodes.emb', 'r')
        counter = 0
        for linen in f:
            if counter != 0:
                line_splt = linen.split()
                nodes[line_splt[0]] = list(line_splt[1:])

            counter += 1

        usr_list = []
        counter = 0
        assert os.path.exists('./saved_models/'+dataset+'/nodes.edgelist')
        lines = []
        f = open('./saved_models/'+dataset+'/nodes.edgelist', 'r')
        for l in f:
            lines.append(l.strip())
        with open('./data/' + dataset + '/files/'+'/all.txt','r') as json_file:
            data = json.load(json_file)
            ID_counter = 0
            for t in data['tweet']:
                s, r = lines[ID_counter].split()
                assert str(t['user_name']) == str(s)
                assert str(t['target_name']) == str(r)
                ID_counter += 1

        f = open('./saved_models/'+dataset+'/nodes.edgelist', 'r')

        for line in f:
            s, r = line.split()
            usr = nodes[s] + nodes[r]
            usr = list(map(float, usr))
            usr_list.append(usr)
            counter += 1


        self.usr_train = usr_list[:num_training]
        self.usr_val = usr_list[num_training:]
        self.usr_all = usr_list

    def get_pretrained_message(self,num_training,num_data,dataset):
        assert os.path.exists('./saved_models/'+ dataset +'/doc2vec_model')
        model_dm = gensim.models.Doc2Vec.load('./saved_models/'+ dataset +'/doc2vec_model')
        doc2vec_list = []
        for ID in range(num_data):
            doc2vec_list.append(model_dm.docvecs['UNSUP_'+str(ID)])


        self.doc2vec_all = doc2vec_list
        self.doc2vec_train = doc2vec_list[:num_training]
        self.doc2vec_val = doc2vec_list[num_training:]

    def get_pretrained_message_removed_sensitives(self,num_training,num_data,dataset):
        assert os.path.exists('./saved_models/'+dataset+'/remove_sensitive_doc2vec_model')
        model_dm = gensim.models.Doc2Vec.load('./saved_models/'+dataset+'/remove_sensitive_doc2vec_model')

        doc2vec_list = []
        doc2vec_remove_list = []
        for ID in range(num_data):
            doc2vec_list.append(model_dm.docvecs['UNSUP_'+str(ID)])
            if 'UNSUP_r_'+str(ID) in model_dm.docvecs:
                doc2vec_remove_list.append(model_dm.docvecs['UNSUP_r_'+str(ID)])
            else:
                doc2vec_remove_list.append(model_dm.docvecs['UNSUP_'+str(ID)])


        assert len(doc2vec_list) == len(doc2vec_remove_list) == num_data

        self.doc2vec_all_removed_sensitives = doc2vec_remove_list
        self.doc2vec_train_removed_sensitives = doc2vec_remove_list[:num_training]
        self.doc2vec_val_removed_sensitives = doc2vec_remove_list[num_training:]

        self.doc2vec_all = doc2vec_list
        self.doc2vec_train = doc2vec_list[:num_training]
        self.doc2vec_val = doc2vec_list[num_training:]

    def get_bounds(self,num_training,dataset):
        assert os.path.exists('./saved_models/'+dataset+'/bounds.txt')
        LB = []
        UB = []
        with open('./saved_models/'+dataset+'/bounds.txt') as json_file:
            bounds_data = json.load(json_file)
            bounds = bounds_data['bounds']
            ID_counter = 0
            for bnd in bounds:
                assert bnd['ID'] == ID_counter
                LB.append(bnd['LB'])
                UB.append(bnd['UB'])
                ID_counter += 1

        self.LB = LB
        self.UB = UB

        self.LB_train = LB[:num_training]
        self.LB_val = LB[num_training:]

        self.UB_train = UB[:num_training]
        self.UB_val = UB[num_training:]