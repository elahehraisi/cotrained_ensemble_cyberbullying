#!/usr/bin/python
import torch.nn as nn
from torch.autograd import Variable
import torch

# ### This script is mainly constructed based on language model in pytorch
# ### https://github.com/pytorch/examples/tree/master/word_language_model
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, is_bn,dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, 1)
        if is_bn == 'bn':
            self.bn = torch.nn.BatchNorm1d(1)
        else:
            self.bn = None


        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = torch.squeeze(torch.mean(output, 0))

        output = self.drop(output)
        if len(output.size()) == 1:
            output = output.view(1,output.size(0))

        decoded = self.decoder(output)

        if self.bn == None:
            return torch.sigmoid(decoded), hidden
        else:
            y_bn = self.bn(decoded)
            return torch.sigmoid(y_bn), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())



class Embedding(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp,pretrained_weight ,is_bn,dropout=0.5,tie_weights=False):
        super(Embedding, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, 1)
        self.for_message = 'True'
        self.is_cuda = False

        if is_bn == 'bn':
            self.bn = torch.nn.BatchNorm1d(1)
        else:
            self.bn = None

        if tie_weights:
            self.decoder.weight = self.encoder.weight
        self.init_weights(pretrained_weight)


    def init_weights(self,pretrained_weight):
        initrange = 0.1
        NoneType = type(None)
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        if type(pretrained_weight) != NoneType:
            self.encoder.weight.data.copy_(pretrained_weight)
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.encoder(input))

        if self.for_message == 'True':
            emb = torch.squeeze(torch.mean(emb, 0))
            if (len(emb.size())) == 1:
                emb = emb.view(1,emb.size()[0])
        else:
            sz = emb.size()
            if len(sz) == 3:
                emb = (emb.view(sz[1], sz[0]* sz[2]))
                self.decoder = nn.Linear(sz[0]* sz[2], 1)

            elif len(sz) == 2:
                emb = (emb.view(1, sz[0] * sz[1]))
                self.decoder = nn.Linear(sz[0] * sz[1], 1)


            if self.is_cuda == True:
                self.decoder = self.decoder.cuda()

        decoded = self.decoder(emb)

        if self.bn == None:
            return torch.sigmoid(decoded)
        else:
            y_bn = self.bn(decoded)
            return torch.sigmoid(y_bn)



class MessageNet(torch.nn.Module):
  def __init__(self,D_in, D_out,is_bn):
      """
      In the constructor we instantiate two nn.Linear modules and assign them as
      member variables.
      """
      super(MessageNet, self).__init__()
      self.linear = torch.nn.Linear(D_in, D_out,bias=True)
      if is_bn == 'bn':
          self.bn = torch.nn.BatchNorm1d(D_out)
      else:
          self.bn = None


  def forward(self, x):
      """
      In the forward function we accept a Variable of input data and we must return
      a Variable of output data. We can use Modules defined in the constructor as
      well as arbitrary operators on Variables.
      """
      tt = self.linear(x)

      if self.bn == None:
        y_pred = torch.sigmoid(tt)
      else:
          tt_bn = self.bn(tt)
          y_pred = torch.sigmoid(tt_bn)

      return y_pred


class UserNet(torch.nn.Module):
    def __init__(self, D_in, D_out,is_bn):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(UserNet, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out,bias=True)
        if is_bn == 'bn':
            self.bn = torch.nn.BatchNorm1d(D_out)
        else:
            self.bn = None

    def forward(self, x):
      """
      In the forward function we accept a Variable of input data and we must return
      a Variable of output data. We can use Modules defined in the constructor as
      well as arbitrary operators on Variables.
      """

      if self.bn == None:
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
      else:
        y_bn = self.bn(self.linear(x))
        y_pred = torch.sigmoid(y_bn)
        return y_pred
