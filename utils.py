#!/usr/bin/python
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gensim
import pickle
import model
import sklearn
import os
import load_data
import torch.nn as nn
import json
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import FeatureHasher

def load_message_learner(args,pretrained):

    if args.message_classifier == 'doc2vec':
        pretrained.get_pretrained_message(args.num_training, args.num_data, args.data_set)
        doc2vec_train = pretrained.doc2vec_train
        doc2vec_val = pretrained.doc2vec_val
        doc2vec_all = pretrained.doc2vec_all
        torch_train = torch.from_numpy(np.array(doc2vec_train))
        torch_val = torch.from_numpy(np.array(doc2vec_val))
        torch_all = torch.from_numpy(np.array(doc2vec_all))

        message_model = model.MessageNet(args.d_in_msg, args.d_out_msg,args.batch_normalization)

        if args.cuda:
            message_model.cuda()

        message_optimizer = args.optimizer(message_model.parameters(), lr=args.lr, weight_decay=0)


        return message_optimizer,message_model, torch_train,torch_val,torch_all

    elif args.message_classifier == 'emb':
        corpus = load_data.Corpus(args.data + args.data_set)
        corpus.get_message_ix_list(args.data + args.data_set)
        train_data_mssg = corpus.train_message_ix
        val_data = corpus.valid_message_ix
        all_data = corpus.all_message_ix

        torch_train = torch.from_numpy(np.array(train_data_mssg))
        torch_val = torch.from_numpy(np.array(val_data))
        torch_all = torch.from_numpy(np.array(all_data))

        pre_init_word = np.array(corpus.get_pretrained_matrix_msg(args.data_set))
        pre_init_word = torch.FloatTensor(pre_init_word.astype(float))

        ntokens = len(corpus.dictionary)
        message_model = model.Embedding(ntokens, args.emsize, pre_init_word, args.batch_normalization,args.dropout, args.tied)

        if args.cuda:
            message_model.cuda()
            pre_init_word.cuda()

        message_optimizer = args.optimizer(message_model.parameters(), lr=args.lr, weight_decay=0)


        return message_optimizer,message_model, torch_train,torch_val,torch_all

    elif args.message_classifier == 'rnn':
        corpus = load_data.Corpus(args.data + args.data_set)
        corpus.get_message_ix_list(args.data + args.data_set)
        train_data_mssg = corpus.train_message_ix
        val_data = corpus.valid_message_ix
        all_data = corpus.all_message_ix

        torch_train = torch.from_numpy(np.array(train_data_mssg))
        torch_val = torch.from_numpy(np.array(val_data))
        torch_all = torch.from_numpy(np.array(all_data))


        ntokens = len(corpus.dictionary)

        message_model = model.RNNModel(args.rnn_model, ntokens, args.emsize, args.nhid, args.nlayers,args.batch_normalization,args.dropout, args.tied)
        if args.cuda:
            message_model.cuda()

        message_optimizer = args.optimizer(message_model.parameters(), lr=args.lr, weight_decay=0)

        return message_optimizer,message_model, torch_train,torch_val,torch_all

    elif args.message_classifier == 'bow':
        corpus = load_data.Corpus(args.data + args.data_set)
        vectorizer = sklearn.feature_extraction.text.HashingVectorizer(n_features=1000)
        messages = vectorizer.fit_transform(corpus.messages_list)
        train_data_mssg = ((messages)[:args.num_training, :])
        val_data = ((messages)[args.num_training:, :])

        args.d_in_msg = (train_data_mssg.shape)[1]

        message_model = model.MessageNet(args.d_in_msg, args.d_out_msg,args.batch_normalization)

        if args.cuda:
            message_model.cuda()

        message_optimizer = args.optimizer(message_model.parameters(), lr=args.lr, weight_decay=0)

        return message_optimizer, message_model, train_data_mssg, val_data, messages

def load_user_learner(args,pretrained):

    if args.user_classifier == 'node2vec':
        pretrained.get_pretrained_user(args.num_training, args.data_set)
        usr_train = pretrained.usr_train
        usr_val = pretrained.usr_val
        usr_all = pretrained.usr_all


        torch_train = torch.FloatTensor(usr_train)
        torch_val = torch.FloatTensor(usr_val)
        torch_all = torch.FloatTensor(usr_all)

        user_model = model.UserNet(args.d_in_user, args.d_out_user,args.batch_normalization)

        if args.cuda:
            user_model.cuda()

        usr_optimizer = args.optimizer(user_model.parameters(), lr=args.lr, weight_decay=0)

        return usr_optimizer,user_model, torch_train,torch_val,torch_all

    elif args.user_classifier == 'emb':
        pretrained.node_tokenize(args.data_set)
        pretrained.get_user_ix(args.num_training, args.data_set)
        pre_init_user = np.array(pretrained.get_pretrained_matrix_usr(args.data_set))
        pre_init_user = torch.FloatTensor(pre_init_user.astype(float))
        usr_train = pretrained.train_nodes_ix
        usr_val = pretrained.val_nodes_ix
        usr_all = pretrained.all_nodes_ix

        torch_train = torch.from_numpy(np.array((usr_train)).astype(int))
        torch_val = torch.from_numpy(np.array((usr_val)).astype(int))
        torch_all = torch.from_numpy(np.array((usr_all)).astype(int))

        n_users = pretrained.num_users_all
        user_model = model.Embedding(n_users, args.useremsize, pre_init_user, args.batch_normalization,args.dropout, args.tied)
        user_model.for_message = 'False'

        if args.cuda:
            user_model.is_cuda = True
            user_model.cuda()
            pre_init_user.cuda()

        usr_optimizer = args.optimizer(user_model.parameters(), lr=args.lr, weight_decay=0)

        return usr_optimizer,user_model, torch_train,torch_val,torch_all

    elif args.user_classifier == 'none':
        return None,None,None,None, None

def find_latest_ckp(load_dir,learner):
    last_chk = -np.inf
    for fl in os.listdir(load_dir):
        if fl.endswith('pth.tar'):
            fl_splt = fl.split('_')
            if fl_splt[0] == learner:
                if fl_splt[2] != 'model.pth.tar':
                    nm = int(fl_splt[2])
                    if nm > last_chk:
                    # nm = fl_splt[2]
                    # if nm == load_epoch:
                        last_chk = nm
                        last_ck_file = fl


    return last_ck_file,last_chk

def get_co_train_loss(y_mssg,y_usr):
    y_mssg_usr = y_mssg[0] - y_usr[0]
    y_pwr = torch.pow(y_mssg_usr, 2)
    return torch.squeeze(0.5 * y_pwr)

def get_weak_loss(y_mssg, LB, UB ,cuda):

    if cuda:
        var_ub = Variable(UB.cuda())
        var_lb = Variable(LB.cuda())
    else:
        var_ub = Variable(UB)
        var_lb = Variable(LB)

    # ######## when we had weak supervision for message classifier only # # -log(min(1.0, 1 + u - y_m)) - log(min(1.0, 1 + y_m - l))
    counter = 0
    for y in y_mssg:

        if cuda:
            b = Variable(torch.ones(y.size()[0]).cuda())
            ones = Variable(torch.ones(y.size()[0]).cuda())
        else:
            b = Variable(torch.ones(y.size()[0]))
            ones = Variable(torch.ones(y.size()[0]))


        a = ones + var_ub - torch.squeeze(y)
        min_ub = torch.log(torch.min(a,b))

        a = ones + torch.squeeze(y) - var_lb
        min_lb = torch.log(torch.min(a,b))


        if counter == 0:
            weak_loss = - min_ub - min_lb
        else:
            weak_loss = weak_loss - min_ub - min_lb

        counter += 1

    weak_loss = torch.squeeze(weak_loss)
    return weak_loss

def cross_entropy_criterion(y_mssg, y_usr,  LB, UB,cuda):
    if len(y_usr) != 0:
        co_trian_loss = get_co_train_loss(y_mssg,y_usr)
        weak_loss = get_weak_loss(y_mssg, LB, UB,cuda)
        loss = torch.add(co_trian_loss , weak_loss)
        total_loss_mean = torch.mean(loss)
        cotrain_loss_mean = torch.mean(co_trian_loss)
        weak_loss_mean = torch.mean(weak_loss)

        if cuda:
            total_loss_mean = total_loss_mean.cpu()
            cotrain_loss_mean = cotrain_loss_mean.cpu()
            weak_loss_mean = weak_loss_mean.cpu()

        return total_loss_mean, cotrain_loss_mean.data.numpy()[0], weak_loss_mean.data.numpy()[0]

    else:
        weak_loss = get_weak_loss(y_mssg, LB, UB,cuda)
        loss = weak_loss
        total_loss_mean = torch.mean(loss)
        weak_loss_mean = torch.mean(weak_loss)
        if cuda:
            total_loss_mean = total_loss_mean.cpu()
            weak_loss_mean = weak_loss_mean.cpu()

        return total_loss_mean, 0, weak_loss_mean.data.numpy()[0]

def reset_grad(param):
    for p in param:
        p.grad.data.zero_()

def labelize_mesages(messages, label_type):
    LabeledSentence = gensim.models.doc2vec.LabeledSentence
    labelized = []
    for i,v in (messages):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def plot_save(J,J_cotrain,J_weak,validation_data,save_dir):

    if len(J_cotrain) != 0:
        plt.figure()
        f, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(J)
        ax1.set_title(validation_data + ' total loss')
        ax2.plot(J_cotrain)
        ax2.set_title(' co-train loss')
        ax3.plot(J_weak)
        ax3.set_title(' weak loss')


        f_loss = open(save_dir + '/' + validation_data + '_total_loss.txt', 'wb')
        pickle.dump(J, f_loss)

        f_loss = open(save_dir + '/' + validation_data + '_cotrain_loss.txt', 'wb')
        pickle.dump(J_cotrain, f_loss)

        f_loss = open(save_dir + '/' + validation_data + '_weak_loss.txt', 'wb')
        pickle.dump(J_weak, f_loss)

        plt.savefig(save_dir + '/loss_' + validation_data + '.png')
        plt.close()
    else:
        plt.figure()
        f, (ax1, ax3,ax4) = plt.subplots(3)
        ax1.plot(J)
        ax1.set_title(validation_data + ' total loss')
        ax3.plot(J_weak)
        ax3.set_title(' weak loss')

        plt.savefig(save_dir + '/loss_' + validation_data + '.png')
        plt.close()

def zero_pad(list_of_list):
    max_len = 0
    for lst in list_of_list:
        if len(lst) > max_len:
            max_len = len(lst)
    for lst in list_of_list:
        if len(lst) < max_len:
            index = 0
            for i in range(max_len - len(lst)):
                lst.insert(index,0)
                index += 1
    return list_of_list

def load_saved_message_learner(args,pretrained,load_dir):

    if args.message_classifier == 'doc2vec':

        pretrained.get_pretrained_message(args.num_training, args.num_data, args.data_set)
        doc2vec_all = pretrained.doc2vec_all
        torch_all = torch.from_numpy(np.array(doc2vec_all))

        message_model = model.MessageNet(args.d_in_msg, args.d_out_msg,args.batch_normalization)

        if args.cuda:
            message_model.cuda()

        msg_lst_ck, selected_chk = find_latest_ckp(load_dir, 'mssage')
        msg_chk = torch.load(load_dir + msg_lst_ck, map_location=lambda storage, location: storage)
        message_model.load_state_dict(msg_chk['state_dict'])

        print('loaded pretrained message learner at ' + str(selected_chk))

        return message_model, torch_all

    elif args.message_classifier == 'emb':
        corpus = load_data.Corpus(args.data + args.data_set)
        corpus.get_message_ix_list(args.data + args.data_set)
        all_data = corpus.all_message_ix
        torch_all = torch.from_numpy(np.array(all_data))
        ntokens = len(corpus.dictionary)
        message_model = model.Embedding(ntokens, args.emsize, None, args.batch_normalization,args.dropout, args.tied)

        if args.cuda:
            message_model.cuda()

        msg_lst_ck, selected_chk = find_latest_ckp(load_dir, 'mssage')
        msg_chk = torch.load(load_dir + msg_lst_ck, map_location=lambda storage, location: storage)
        message_model.load_state_dict(msg_chk['state_dict'])


        return message_model, torch_all

    elif args.message_classifier == 'rnn':
        corpus = load_data.Corpus(args.data + args.data_set)
        corpus.get_message_ix_list(args.data + args.data_set)

        all_data = corpus.all_message_ix
        torch_all = torch.from_numpy(np.array(all_data))
        ntokens = len(corpus.dictionary)
        message_model = model.RNNModel(args.rnn_model, ntokens, args.emsize, args.nhid, args.nlayers,args.batch_normalization,
                                       args.dropout, args.tied)
        if args.cuda:
            message_model.cuda()

        msg_lst_ck, selected_chk = find_latest_ckp(load_dir, 'mssage')
        msg_chk = torch.load(load_dir + msg_lst_ck, map_location=lambda storage, location: storage)
        message_model.load_state_dict(msg_chk['state_dict'])

        return message_model, torch_all

    elif args.message_classifier == 'bow':
        corpus = load_data.Corpus(args.data + args.data_set)
        vectorizer = sklearn.feature_extraction.text.HashingVectorizer(n_features=1000)
        messages = vectorizer.transform(corpus.messages_list)
        train_data_mssg = ((messages)[:args.num_training, :])

        args.d_in_msg = (train_data_mssg.shape)[1]

        torch_all = messages

        message_model = model.MessageNet(args.d_in_msg, args.d_out_msg,args.batch_normalization)

        if args.cuda:
            message_model.cuda()

        msg_lst_ck, selected_chk = find_latest_ckp(load_dir, 'mssage')
        msg_chk = torch.load(load_dir + msg_lst_ck, map_location=lambda storage, loc: storage)
        message_model.load_state_dict(msg_chk['state_dict'])


        return message_model, torch_all

def load_saved_user_learner(args,pretrained,load_dir):

    if args.user_classifier == 'node2vec':
        pretrained.get_pretrained_user(args.num_training, args.data_set)
        usr_all = pretrained.usr_all

        torch_all = torch.FloatTensor(usr_all)
        user_model = model.UserNet(args.d_in_user, args.d_out_user,args.batch_normalization)

        if args.cuda:
            user_model.cuda()

        usr_lst_ck, selected_chk = find_latest_ckp(load_dir,'user')
        usr_chk = torch.load(load_dir + usr_lst_ck, map_location=lambda storage, location: storage)
        user_model.load_state_dict(usr_chk['state_dict'])

        print('loaded node2vec user learner at ' + str(selected_chk))

    elif args.user_classifier == 'emb':
        pretrained.node_tokenize(args.data_set)
        pretrained.get_user_ix(args.num_training, args.data_set)
        usr_all = pretrained.all_nodes_ix

        torch_all = torch.from_numpy(np.array((usr_all)).astype(int))

        n_users = pretrained.num_users_all
        user_model = model.Embedding(n_users, args.useremsize, None, args.batch_normalization, args.dropout, args.tied)
        user_model.for_message = 'False'
        user_model.decoder = nn.Linear(args.useremsize * 2, 1)

        if args.cuda:
            user_model.is_cuda = True
            user_model.cuda()

        usr_lst_ck, selected_chk = find_latest_ckp(load_dir, 'user')
        usr_chk = torch.load(load_dir + usr_lst_ck, map_location=lambda storage, location: storage)
        user_model.load_state_dict(usr_chk['state_dict'])

    elif args.user_classifier == 'none':
        return None,None

    return user_model,torch_all

def get_top_user_pairs(args,score,save_dir):
    f_user = open('./saved_models/' + args.data_set + '/' + 'nodes.edgelist', 'r')

    bully = []
    victim = []
    with open('./data/' + args.data_set + '/files/' + '/all.txt', 'r') as json_file:
        data = json.load(json_file)
        for p in data['tweet']:
            bully.append(p['user_name'])
            victim.append(p['target_name'])

    # # # ###########******** sanity check to see if users in nodes.edgelist.txt are the same as all file
    i = 0
    for line in f_user:
        line = line.strip()
        ln_spl = line.split()
        b = str(ln_spl[0].strip())
        v = str(ln_spl[1].strip())
        assert b == str(bully[i])
        assert v == str(victim[i])
        i += 1
    # #
    score = list(score.data)
    bully = bully[:len(score)]
    victim = victim[:len(score)]

    dic_bully = {}
    dic_victim = {}
    bully_list = list(bully)
    victim_list = list(victim)

    for i, b_l in enumerate(bully_list):
        dic_bully.setdefault(b_l, []).append(i)

    for i, v_l in enumerate(victim_list):
        dic_victim.setdefault(v_l, []).append(i)
    #
    print ('start computing the average bullying score for user pair...')
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
            # the minimum number of interaction is conversation here is 2. But you could change if
            if len(shared) >= 2:
                avr = 0
                for j in shared:
                    avr += score[j]

                if args.cuda:
                    avr = avr.cpu()
                dic_u[str(b_u) + ',' + str(v_u)] = np.true_divide(avr.numpy(), len(shared))

            selected.add((b_u, v_u))
            selected.add((v_u, b_u))

    sorted_dic = sorted(dic_u.items(), key=operator.itemgetter(1), reverse=True)
    f_w = open(save_dir + '/user_pair_indices.txt', 'w')

    for key in sorted_dic:
        f_w.write(str(key[0]) + ',' + str(key[1]))
        f_w.write('\n')
