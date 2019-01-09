#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import argparse
import torch
from torch.autograd import Variable
import utils
import load_data
import numpy as np
import os
import json
import scipy


# this script is the implementation of co-trained ensemble
# for more information please read the paper "Weakly Supervised Cyberbullying Detection usingCo-trained Ensembles of Embedding Models"
# http://people.cs.vt.edu/~bhuang/papers/raisi-asonam18.pdf

# #### Please run all preprocess.py script before running this script

parser = argparse.ArgumentParser(description='co-traine ensemble models for cyberbullying detection')
parser.add_argument('--data_set', type=str, default='synthetic',
                     help='datasets: synthetic/Twitter/Instagram')
parser.add_argument('--data', type=str, default='./data/',
                    help='location of the data corpus')
parser.add_argument('--rnn_model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--useremsize', type=int, default=100,
                    help='size of user embeddings')
parser.add_argument('--checkpoint', type=int, default=5,
                    help='define the point we check the validation set to save the model')
parser.add_argument('--patient', type=int, default=2,
                    help='the number of times we will pursue training even if validation error is increasing')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=101,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout) 0.2')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--d_in_user', type=int, default=200,
                    help='input dimension for users; it is created based on node2vec')
parser.add_argument('--d_out_user', type=int, default=1,
                    help='output dimension for users; it is one since we have binary classifier')
parser.add_argument('--optimizer', default=torch.optim.Adagrad,
                    help='optimizer type')
parser.add_argument('--num_training', type=int,
                    help='number of training')
parser.add_argument('--num_validation', type=int,
                    help='number of validation')
parser.add_argument('--num_data', type=int,
                    help='total number of data')
parser.add_argument('--d_in_msg', type=int, default=100,
                    help='input dimension for message')
parser.add_argument('--d_out_msg', type=int, default=1,
                    help='output dimension for message')
parser.add_argument('--message_classifier', type=str, default='rnn',
                    help='doc2vec/rnn/emb/bow')
parser.add_argument('--user_classifier', type=str, default='none',
                    help='node2vec/emb/none')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--batch_normalization', type=str, default='bn',
                    help='bn/no-bn  if bn: batch normalization layer is used. if no-bn: no batch normalization is used')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# ####################################################################
# ######**************** get the number of training/test/all data
# ####################################################################

with open('./data/' + args.data_set + '/files/'+'training.txt','r') as json_file:
    data = json.load(json_file)
    args.num_training = (len(data['tweet']))
    print('number of training: %d' % args.num_training)

with open('./data/' + args.data_set + '/files/'+ 'validation.txt', 'r') as json_file:
    data = json.load(json_file)
    args.num_validation = (len(data['tweet']))
    print('number of validation: %d' % args.num_validation)

with open('./data/' + args.data_set + '/files/'+ 'all.txt', 'r') as json_file:
    data = json.load(json_file)
    args.num_data = (len(data['tweet']))
    print('total number of data: %d' % args.num_data)

assert args.num_validation+ args.num_training == args.num_data

# ####################################################################
# ######**************** load pre-computed upper and lower bounds for each message
# ####################################################################

pretrained = load_data.Pretrained()
pretrained.get_bounds(args.num_training, args.data_set)
LB_train = pretrained.LB_train
LB_val = pretrained.LB_val
UB_train = pretrained.UB_train
UB_val = pretrained.UB_val
LB_all = pretrained.LB
UB_all = pretrained.UB

assert len(LB_train)+len(LB_val) == args.num_training+ args.num_validation

if args.dropout == 0:
    dropout = '0'
else:
    dropout = str(args.dropout)

# ####################################################################
# ######**************** create the message and user learner models
# ####################################################################

message_optimizer,message_model,train_message,val_message,all_message = utils.load_message_learner(args,pretrained)
user_optimizer,user_model,train_user,val_user,all_user = utils.load_user_learner(args, pretrained)

# ####################################################################
# ######**************** determine the directory to save the models during training
# ####################################################################

save_dir = './data/' + args.data_set + '/saved_data/' + args.message_classifier + '_' + args.user_classifier + '/'+ args.batch_normalization + '/dropout_' + dropout + '/'


print('save directory: '+save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(data,indx):
    if scipy.sparse.issparse(data)== True :
        crs_dt = data[indx * args.batch_size: indx * args.batch_size + args.batch_size,:]
        return torch.from_numpy(crs_dt.A).float()

    else:
        return data[indx * args.batch_size: indx * args.batch_size + args.batch_size]


def train():
    # tell PyTorch to use training mode (dropout, batch norm, etc)
    message_model.train()
    if args.user_classifier != 'none':
        user_model.train()


    if args.message_classifier == 'rnn':
        hidden = message_model.init_hidden(args.batch_size)

    iter = int(args.num_training / args.batch_size)
    for indx in range(iter):
        y_mssg = []
        y_usr = []

        x_message = Variable(get_batch(train_message,indx))

        if args.user_classifier != 'none':
            x_user = Variable(get_batch(train_user,indx))

        if args.cuda:
            x_message = x_message.cuda()
            if args.user_classifier != 'none':
                x_user = x_user.cuda()



        if (x_message.size())[0] == args.batch_size:
            ## Update message learner parameters
            if args.message_classifier == 'rnn':
                hidden1 = repackage_hidden(hidden)
                y_msg_rnn, hidden = message_model(x_message.t(), hidden1)
                y_mssg.append(y_msg_rnn)

            elif args.message_classifier == 'emb':
                y_msg_emd = message_model(x_message.t())
                y_mssg.append(y_msg_emd)

            else:
                y_msg = message_model(x_message)
                y_mssg.append(y_msg)


            if args.user_classifier == 'emb':
                y_user = user_model(x_user.t())
                y_usr = [y_user]
            elif args.user_classifier == 'node2vec':
                y_user = user_model(x_user)
                y_usr = [y_user]



            lb = torch.FloatTensor(get_batch(LB_train,indx))
            ub = torch.FloatTensor(get_batch(UB_train, indx))

            loss, _, _ = utils.cross_entropy_criterion(y_mssg, y_usr, lb, ub, args.cuda)


            loss.backward(retain_variables=True)

            message_optimizer.step()

            utils.reset_grad(message_model.parameters())
            if args.user_classifier != 'none':
                utils.reset_grad(user_model.parameters())


            ## Update user learner parameters if there is user learner
            if args.user_classifier != 'none':

                if args.message_classifier == 'doc2vec' or args.message_classifier == 'bow':
                    y_msg = message_model(x_message)
                    y_mssg = [y_msg]

                elif args.message_classifier == 'rnn':
                    hidden1 = repackage_hidden(hidden)
                    y_msg, hidden = message_model(x_message.t(), hidden1)
                    y_mssg = [y_msg]


                elif args.message_classifier == 'emb':
                    y_msg = message_model(x_message.t())
                    y_mssg = [y_msg]


                if args.user_classifier == 'emb':
                    y_user = user_model(x_user.t())
                    y_usr = [y_user]
                else:
                    y_user = user_model(x_user)
                    y_usr = [y_user]


                loss, _, _ = utils.cross_entropy_criterion(y_mssg, y_usr, lb, ub, args.cuda)


                loss.backward()

                user_optimizer.step()

                utils.reset_grad(message_model.parameters())
                utils.reset_grad(user_model.parameters())

def validate(validation_data,save_score=False):
    # Turn on evaluation mode which disables dropout.
    message_model.eval()
    if args.user_classifier != 'none':
        user_model.eval()

    if args.message_classifier == 'rnn':
        hidden = message_model.init_hidden(args.batch_size)


    if validation_data == 'train':
        message_for_validation = train_message
        user_for_validation = train_user
        iter = int(args.num_training / args.batch_size)
        lb_for_validation = LB_train
        ub_for_validation = UB_train


    elif validation_data == 'validation':
        message_for_validation = val_message
        user_for_validation = val_user
        iter = int(args.num_validation / args.batch_size)
        lb_for_validation = LB_val
        ub_for_validation = UB_val


    elif validation_data == 'all':
        message_for_validation = all_message
        user_for_validation = all_user
        iter = int(args.num_data / args.batch_size)
        lb_for_validation = LB_all
        ub_for_validation = UB_all


    weak_loss = 0
    total_loss = 0
    co_train_loss = 0
    num_elements = 0
    for indx in range(iter):
        y_mssg = []
        y_usr = []

        x_message = Variable(get_batch(message_for_validation, indx))


        if args.user_classifier != 'none':
            x_user = Variable(get_batch(user_for_validation,indx))

        x_message.volatile = True
        if args.user_classifier != 'none':
            x_user.volatile = True

        if (x_message.size())[0] == args.batch_size:
            if args.cuda:
                x_message = x_message.cuda()
                if args.user_classifier != 'none':
                    x_user = x_user.cuda()


            if args.message_classifier == 'rnn':
                hidden1 = repackage_hidden(hidden)
                y_msg_rnn, hidden = message_model(x_message.t(), hidden1)
                y_mssg.append(y_msg_rnn)


            elif args.message_classifier == 'emb':
                y_msg_emd = message_model(x_message.t())
                y_mssg.append(y_msg_emd)

            else:
                y_msg = message_model(x_message)
                y_mssg.append(y_msg)


            if args.user_classifier == 'emb':
                y_user = user_model(x_user.t())
                y_usr = [y_user]
            elif args.user_classifier == 'node2vec':
                y_user = user_model(x_user)
                y_usr = [y_user]


            if save_score == True:
                if indx == 0:
                    if args.user_classifier != 'none':
                        score = torch.div(torch.add(y_usr[0], y_mssg[0]), 2)
                    else:
                        score = y_mssg[0]

                else:
                    if args.user_classifier != 'none':
                        score = torch.cat((score, torch.div(torch.add(y_usr[0], y_mssg[0]), 2)))
                    else:
                        score = torch.cat((score, y_mssg[0]))



            lb = torch.FloatTensor(get_batch(lb_for_validation, indx))
            ub = torch.FloatTensor(get_batch(ub_for_validation, indx))


            loss, co_loss, wk_loss = utils.cross_entropy_criterion(y_mssg, y_usr, lb, ub,args.cuda)


            assert len(x_message) <= args.batch_size
            total_loss += loss * len(x_message)
            co_train_loss += (co_loss) * len(x_message)
            weak_loss += (wk_loss) * len(x_message)

            num_elements += len(x_message)

    validation_loss = {}
    if save_score == True:
        validation_loss['score'] = score
    validation_loss['total_loss'] = total_loss[0]/ num_elements
    validation_loss['cotrain_loss'] = co_train_loss / num_elements
    validation_loss['weak_loss'] = weak_loss / num_elements


    return validation_loss

def save_checkpoint(state,save_dir, filename):
    torch.save(state, save_dir+filename+'_model.pth.tar')

def save_models(n_epoch):
    msg_save_mdl = message_model
    msg_save_opt = message_optimizer

    user_save_mdl = user_model
    user_save_opt = user_optimizer

    save_checkpoint({
        'epoch': n_epoch,
        'state_dict': msg_save_mdl.state_dict(),
        'optimizer': msg_save_opt.state_dict(),
    },save_dir,'mssage_saved_'+str(n_epoch))

    if args.user_classifier != 'none':
        save_checkpoint({
            'epoch': n_epoch,
            'state_dict': user_save_mdl.state_dict(),
            'optimizer': user_save_opt.state_dict(),
        },save_dir,'user_saved_'+str(n_epoch))


# ####################################################################
# ######**************** start training and validating models
# ####################################################################
try:
    J = []
    J_cotrain = []
    J_weak = []

    J_val = []
    J_cotrain_val = []
    J_weak_val = []

    J_all = []
    J_all_cotrain = []
    J_all_weak = []

    patient_counter = 0
    is_saved = False
    lr = args.lr
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        print ('epoch %d' % epoch)
        train()
        validation_loss = validate('train')
        total_loss = validation_loss['total_loss']
        cotrain_loss = validation_loss['cotrain_loss']
        weak_loss = validation_loss['weak_loss']

        J.append(total_loss.data.numpy()[0])
        J_cotrain.append(cotrain_loss)
        J_weak.append(weak_loss)

        validation_loss = validate('validation')
        total_loss_val = validation_loss['total_loss']
        cotrain_loss_val = validation_loss['cotrain_loss']
        weak_loss_val = validation_loss['weak_loss']
        J_val.append(total_loss_val.data.numpy()[0])
        J_cotrain_val.append(cotrain_loss_val)
        J_weak_val.append(weak_loss_val)


        validation_loss = validate('all',save_score=True)
        score = validation_loss['score']
        total_loss_all = validation_loss['total_loss']
        cotrain_loss_all = validation_loss['cotrain_loss']
        weak_loss_all = validation_loss['weak_loss']

        J_all.append(total_loss_all.data.numpy()[0])
        J_all_cotrain.append(cotrain_loss_all)
        J_all_weak.append(weak_loss_all)

        if epoch % args.checkpoint == 0:
            if total_loss_val.data[0] < best_val_loss:
                best_val_score = score
                if best_val_loss == np.inf:
                    best_val_loss = total_loss_val.data[0]
                else:
                    best_val_loss = total_loss_val.data[0]

                    utils.plot_save(J, J_cotrain, J_weak, 'training_' + str(epoch), save_dir)
                    utils.plot_save(J_val, J_cotrain_val, J_weak_val, 'validation_' + str(epoch), save_dir)
                    utils.plot_save(J_all, J_all_cotrain, J_all_weak,'all_' + str(epoch), save_dir)
                    save_models(epoch)
                    is_saved = True
                    patient_counter = 0


            else:
                patient_counter += 1


                if patient_counter > args.patient:
                    print('stop training at epoch: ' + str(epoch))
                    break


    if is_saved == False: ##### when validation loss is increasing from beginning, so no single model is saved so far
        save_models(epoch)
        best_val_score = score

        utils.plot_save(J, J_cotrain, J_weak, 'training_' + str(epoch), save_dir)
        utils.plot_save(J_val, J_cotrain_val, J_weak_val, 'validation_' + str(epoch), save_dir)
        utils.plot_save(J_all, J_all_cotrain, J_all_weak,'all_' + str(epoch), save_dir)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')



