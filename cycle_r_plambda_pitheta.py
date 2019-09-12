import argparse 
import time 
from datetime import datetime
import os

import sqlite3
import random
from random import shuffle
import math
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
from fuzzysearch import find_near_matches

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
parser.add_argument('--epochs', type=int, default=130, help='maximum number of epochs')
parser.add_argument('--ds_size', type=int, default=1000, help='training set size')
parser.add_argument('--distill_size', type=int, default=20000, help='training set size')
parser.add_argument('--motif', type=int, default=2, help='=1= short motif, =4= long motif')
parser.add_argument('--nmotifs', type=int, default=1, help='number of motifs that define the process')
parser.add_argument('--mtype', type=str, default='m', help='m, mam, m1m2')
parser.add_argument('--n', type=int, default=30, help='string size')
parser.add_argument('--p', type=float, default=0.5, help='probability of flipping a coin')
parser.add_argument('--print_softm', type=str, default='', help='train or print')
parser.add_argument('--job', type=int, default=0, help='slurm job id')

#parser.add_argument('--feat', type=str, default='111', help='features for motifs with -.- separator; 0 or 1 at i-th position adds 0 to motif')
#parser.add_argument('--feat', type=str, default='1101000', help='features for motifs with -.- separator; (motif, supermotif, submotif, 1st bit==0, 10101, 1001001, 00110011)')
parser.add_argument('--feat', type=str, default='1011111', help='features for motifs with -.- separator; (motif, supermotif, submotif__2, 1st bit==0, 10101_len_m, 1001001_le_m_2, 00110011_len_m__2)')
parser.add_argument('--train', type=str, default='rs', help='=rs= rejection sampling, =snis_mix= snis mixture, =snis_r= snis r')
parser.add_argument('--restore', type=str, default='', help='checkpoint to restore model from')
parser.add_argument('--theta_fixed', action='store_false', help='train theta with lambda (log-linear model) or only lambda')
parser.add_argument('--test_run', action='store_true', help='if False - testing run, do not store accuracies')
parser.add_argument('--cyclic', action='store_true', help='if True - no distillation from original r but improved pi_theta')

args = parser.parse_args()

torch.set_printoptions(precision=15)


if args.mtype in ['m', 'mam']:
    args.nmotifs = 1
elif args.mtype == 'm1m2':
    args.nmotifs = 2
    # selector bit
    args.n = args.n + 1

if args.nmotifs == 1:
    assert len(args.feat) == 7
elif args.nmotifs == 2:
    assert len(args.feat) == 8

s = "\nParameters:\n"
for k in sorted(args.__dict__):
    s += "{} = {} \n".format(k.lower(), args.__dict__[k])
print(s)

# input vocabulary size
ntoken = 5
batch_size = 500

nhid = 200

# one hot input vector embeddings size
ninp = 3
nlayers = 2

dropout = 0.2 # prob to be zeroed
loss_scale = 1
log_interval = 10
clip = 0.25
nsamples = 10

start_symbol = torch.tensor([[3]*10000]).cuda()
PAD = 4

timestamp = datetime.now().strftime("%mm%dd_%H%M%S%f")
print(timestamp)

# motif: ratio = 1: 1:50, 2: 1:100, 3: 1:500, 4: 1:1000
# choose 2,4,5,6,7
if args.nmotifs == 1:
    if args.mtype == 'm':
        if args.motif == 1:
            all_motifs = {30:'1000101111', 50:'10001010001'}
            power_motifs = {30:21927961, 50:21571947468791}
        elif args.motif == 2:
            all_motifs = {30:'10001010001', 50:'100010100010'}
            power_motifs = {30:10355564, 50:10547846544409}
        elif args.motif == 3:
            all_motifs = {30:'1000101000101', 50:'10011000111111', 100:'0111010000011101'}
            power_motifs = {30:2334480, 50:2541261794559}
        elif args.motif == 4:
            all_motifs = {30:'10001011111000', 50:'100110001111111'}
            power_motifs = {30:1113640, 50:1236662229247}
        elif args.motif == 5:
            all_motifs = {30:'01011101101'}
        elif args.motif == 6:
            all_motifs = {30:'001001100111'}
        elif args.motif == 7:
            all_motifs = {30:'1011100111001'}
    elif args.mtype == 'mam':
        if args.motif == 2:
            all_motifs = {30:'100010100011.100010100011'}
            power_motifs = {30:11787265}
        elif args.motif == 3:
            all_motifs = {30:'10001011111000.10001011111000'}
            power_motifs = {30:3064058}
        elif args.motif == 4:
            all_motifs = {30:'1000101111100011.1000101111100011'}
            power_motifs = {30:786542}
        elif args.motif == 5:
            all_motifs = {30:'01011101101.01011101101'}
        elif args.motif == 6:
            all_motifs = {30:'001001100111.001001100111'}
        elif args.motif == 7:
            all_motifs = {30:'1011100111001.1011100111001'}
elif args.nmotifs == 2:
    if args.motif == 2:
        all_motifs = {31:'100010111110.011101000001'}
        power_motifs = {31:9953280}
    elif args.motif == 3:
        all_motifs = {31:'10001011111000.01110100000111'}
        power_motifs = {31:2227280}
    elif args.motif == 4:
        all_motifs = {31:'100010111110001.011101000001110'}
        power_motifs = {31:1048182}
        
entp_motifs_tm = {30:{'mam.100010100011':16.282530254126048/31, 'mam.1000101111100011':13.57540128031525/31,
                    'm.10001010001':16.15303451776991/31,'m.10001011111000':13.923144487457433/31,
                 'mam.10001011111000':14.935250784153713/31, 'm.01011101101':16.1633538708637/31, 
                 'm.001001100111':15.420728378322668/31,'m.1011100111001':14.6736907/31, 'mam.01011101101':16.950563779/31,
                 'mam.001001100111':16.2827152768/31, 'mam.1011100111001':15.61062622/31, 'm.1000101000101':14.66329972621143/31}, 100:{'m.0111010000011101':62.665668876452344/101}}

z_motifs = {30:{'mam.100010100011':0.0046360, 'mam.1000101111100011':0.00022888,
                    'm.10001010001':0.00964437,'m.10001011111000':0.0010371580,
                 'mam.10001011111000':0.001037158, 'm.01011101101':0.0097444, 
                 'm.001001100111':0.004637, 'm.1011100111001':0.002196863, 'mam.01011101101':0.00974440,
                 'mam.001001100111':0.004637, 'mam.1011100111001':0.002196863, 'm.1000101000101':0.0021741539239883423}, 100:{'m.0111010000011101':0.0012952530732785747}}

entp_motifs = {}

for ni, m in all_motifs.items():
    if ni in entp_motifs_tm:
        entp_motifs[ni] = entp_motifs_tm[ni][args.mtype+'.'+m.split('.')[0]]


# data generation

def get_batch(source_data, batch):
    data = source_data[:-1,batch:batch+batch_size]
    target = source_data[1:,batch:batch+batch_size].contiguous().view(-1)
    return data, target

def get_batch_fsz(source_data, batch):
    data = source_data[:-1,batch:batch+batch_size]
    target = source_data[1:,batch:batch+batch_size].contiguous()
    return data, target

def load_data_motif(n, sz, motif, ds_type):
    ds = ""
    # input: <bos> binary string <eos>
    # 3 {0,1}^n 2
    if args.nmotifs == 1:
        data_file = os.path.join('/tmp-network/user/tparshak/data', 'pfsa_%d_%s'%(n, motif),"%s.txt"%ds_type)
    else:
        data_file = os.path.join('/tmp-network/user/tparshak/data', 'pfsa_%d_%s'%(n-1, motif),"%s.txt"%ds_type)
    
    with open(data_file, "r") as file:
        for line in file:
            #assert motif in line
            ds += line.strip()
            #print(line.strip())
            if len(ds)>=sz*n:
                break
    original = ''.join(c+' ' for c in ds[:sz*n]).strip()
    original = np.fromstring(original, dtype=int, sep=' ')
    original = original.reshape((original.shape[0]//n, n)).transpose()
    
    for i in range(original.shape[1]):
        res = ''.join([str(original[j,i]) for j in range(original.shape[0])])
        #assert flag
    
    dataset = (np.ones((n+2, original.shape[1]))).astype(int)
    dataset[1:-1] = original
    dataset[0] = dataset[0]*3
    dataset[-1] = dataset[-1]*2
    print(dataset.shape, batch_size)
    assert dataset.shape[1] >= sz
    ds = dataset[:, :batch_size*int(1.0*dataset.shape[1]/batch_size)]
    return torch.from_numpy(ds).cuda()

# model

# language model architecture from 
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModel/
class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        
        self.encoder = nn.Embedding(ntoken, ninp)
        one_hot_vecs = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0], [0,0,0]])
        self.encoder.weight.data.copy_(torch.from_numpy(one_hot_vecs))
        self.freeze_layer(self.encoder)
        
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout) 
        # <bos> is not in the output vocabulary       
        self.decoder = nn.Linear(nhid, ninp)

        self.init_weights()
        
        self.nhid = nhid
        self.nlayers = nlayers
    
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, len_inp, mask):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output) # [seq_len ,batch, nhid]
        # [seq_len*batch, ntok]
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = torch.mul(decoded.view(output.size(0), output.size(1), decoded.size(1)), mask)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

def oracle_features(s, motifs, feat):
    # s: seq_len x 1
    # output: [ nfeat ]
    # (motif, supermotif, submotif__2, 1st bit==0, 10101_len_m, 1001001_le_m_2, 00110011_len_m__2)
    out = []
    idx = min(1, len(s)-1)
    nfeat = sum([sum([int(e!='0') for e in el]) for el in feat])
    i = 0
    for j in range(len(feat[i])):
        if feat[i][j] == '1':
            if j < len(args.feat)-4:
                if args.nmotifs == 1:
                    if j==0:
                        # motif
                        out += [1 - int(motifs[i] in s)]
                    elif j==1:
                        # supermotif
                        motif_j = motifs[i] + '0'*1
                        out += [1 - int(motif_j in s)]
                    elif j==2:
                        # submotif
                        motif_j = motifs[i][:len(motifs[i])//2]
                        out += [1 - int(motif_j in s)]
                elif args.nmotifs == 2:
                    if j in [0,2]:
                        # motif
                        out += [1 - int(motifs[max(0, j-1)] in s)]
                    elif j in [1,3]:
                        # submotif
                        motif_j = motifs[max(0, j-2)][:len(motifs[max(0, j-2)])//2]
                        out += [1 - int(motif_j in s)]
            else:
                if j == len(args.feat)-4:
                        # first bit
                        out += [1 - int(s[idx]=='1')]
                # distractor
                elif j == len(args.feat)-3:
                    pref = '10101'
                    motif_j = (pref*args.n)[:len(motifs[i])]
                    out += [1 - int(motif_j in s)]
                elif j == len(args.feat)-2:
                    pref = '1001001'
                    motif_j = (pref*args.n)[:len(motifs[i])+2]
                    out += [1 - int(motif_j in s)]
                elif j == len(args.feat)-1:
                    pref = '00110011'
                    motif_j = (pref*args.n)[:len(motifs[i])//2]
                    out += [1 - int(motif_j in s)]
        elif feat[i][j] == 'e':
            # edit distance
            #mdist = find_near_matches(motifs[i], s, max_l_dist=1)
            #if mdist:
            #    dist = mdist[0].dist
            #else:
            #    dist=len(motifs[i])
            #out += [(dist*1.0)/len(motifs[i])]
            out += [get_edit_frc(s, motifs[i])]
        elif feat[i][j] == 's':
            out += [get_longestsubstr_frc(s, motifs[i])]
        elif feat[i][j] == 'm':
            out += [get_longestsubstr_frc(s, motifs[i]) + get_edit_frc(s, motifs[i])]
    return out


def get_longestsubstr_frc(s, motif):
    
    n, m = len(s)+1, len(motif)+1
    #assert m<=n
    e = np.zeros((m,n))
    max_lss = 0
    for j in range(1,m):
        for i in range(1,n):
            e_ij = []
            if s[i-1]!=motif[j-1]:
                e[j,i]=0
            else: 
                e[j,i] = 1 + e[j-1,i-1]
            max_lss = max(max_lss, e[j,i])
    return 1-max_lss/len(motif)

def get_edit_frc(s, motif):
    def edit_distance(subs, motif):
        n, m = len(subs)+1, len(motif)+1
        #assert m<=n
        e = np.zeros((m,n))
        e[0,0] = 0
        for j in range(1,m):
            e[j,0]= j
        
        for j in range(1,m):
            for i in range(1,n):
                e_ij = []
                if j>0:
                    e_ij += [e[j-1,i]+1]
                if i>0:
                    e_ij += [e[j,i-1]+1]
                if j>0 and i>0:
                    e_ij += [e[j-1, i-1]+ int(subs[i-1]!=motif[j-1])]
                if e_ij:
                    e[j,i] = min(e_ij)
        return 1.0*min(e[-1,:])
    ed_dist =  edit_distance(s, motif)
    edit_frac = ed_dist/len(motif)
    assert edit_frac<=1
    return edit_frac

def get_edit_frc1(s, motif):
    def edit_distance(subs, motif):
        n, m = len(subs)+1, len(motif)+1
        #assert m<=n
        e = np.ones((m,n))*m
        e[0,0] = 0
        for j in range(1,m):
            e[j,0]=j
        for i in range(1,n):
            e[0,i]=i
        for j in range(1,m):
            for i in range(1,n):
                e_ij = []
                if j>0:
                    e_ij += [e[j-1,i]+1]
                if i>0:
                    e_ij += [e[j,i-1]+1]
                if j>0 and i>0:
                    e_ij += [e[j-1, i-1]+ int(subs[i-1]!=motif[j-1])]
                if e_ij:
                    e[j,i] = min(e_ij)
        return 1.0*min(e[-1,:])
    
    ed_dist = len(motif)
    for i in range(len(s)):
        for j in range(i, len(s)):
            ed_dist = min(ed_dist, edit_distance(s[i:i+j], motif))#
    edit_frac = ed_dist/len(motif)
    assert edit_frac<=1
    return edit_frac

    
def get_features(var, motifs, feat):
    # returns the results of identifying oracle features in the input binary sequence
    # 0 = feature exists
    # var:      [ seq_len x batch ]
    # output:   [batch x nfeat]
    def var_to_str(a):
        a = a.data.cpu().numpy()
        b = []
        for i in range(a.shape[1]):
            b += [''.join([str(el) for el in a[:,i]])]
        return b
    x = var_to_str(var)
    out = []
    for b in x:
        out += [oracle_features(b, motifs, feat)]
    return torch.tensor(out).cuda().float()
    

class GAMModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, feat, motifs, dropout=0.5):
        super(GAMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.feat = feat
        self.motifs = motifs
        
        self.encoder = nn.Embedding(ntoken, ninp)
        # 0   1  <eos>   <bos>   PAD
        one_hot_vecs = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0], [0,0,0]])
        self.encoder.weight.data.copy_(torch.from_numpy(one_hot_vecs))
        self.freeze_layer(self.encoder)
        
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout) 
        # <bos> is not in the output vocabulary       
        self.decoder = nn.Linear(nhid, ninp)

        if args.theta_fixed:
            self.freeze_layer(self.decoder)
            self.freeze_layer(self.rnn)
    
        self.feat = feat
        self.motifs = motifs
        nfeat = sum([sum([int(e!='0') for e in el]) for el in feat])
        self.lin_lambda = nn.Linear(nfeat, 1)
        self.lin_lambda.bias.data = self.lin_lambda.bias.data * 0
        self.lin_lambda.bias.requires_grad = False
        self.lin_lambda.weight.data = self.lin_lambda.weight.data * 0

        self.init_weights()
        
        self.nhid = nhid
        self.nlayers = nlayers
    
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, len_inp, mask):
        emb = self.encoder(input)
        emb_pack = torch.nn.utils.rnn.pack_padded_sequence(emb, len_inp, batch_first=False)
        
        out_pack, hidden = self.rnn(emb_pack, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=False)
        output = torch.mul(output, mask)
        # [seq_len x batch x nhid]
        output = self.drop(output) 
        # [ seq_len*batch x ntok]
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        
        x_feat = get_features(input, self.motifs, self.feat)
        log_lin = self.lin_lambda(x_feat)
        
        decoded = torch.mul(decoded.view(output.size(0), output.size(1), decoded.size(1)), mask)
        
        return decoded, hidden, log_lin

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


def sample_lm(model, batch_size_i):
    # output: [ seq_len x batch ]
    
    # [ 1 x batch ]
    # <pad> idx = 4
    out = [(torch.ones(1)*3).cuda().long()]*batch_size_i # contains sequences of variable lenght
    symb = start_symbol[:,:batch_size_i]
    hidden = model.init_hidden(batch_size_i)
    len_inp = torch.ones((batch_size_i), dtype=torch.int64)
    mask = torch.ones((1, batch_size_i, 1)).cuda()

    for i in range(args.n+1):
        # [1 x batch x ntok]
        logits, hidden = model(symb, hidden, len_inp, mask)[:2]
        probs = softm(logits)
        cat_dist = torch.distributions.Categorical(probs=probs)
        # [ 1 x batch ]
        symb = cat_dist.sample()
        for b in range(batch_size_i):
            if i==0 or (i>0 and out[b][-1] != 2): 
                out[b] = torch.cat((out[b], symb[:1,b]), dim=0)
    out = torch.nn.utils.rnn.pad_sequence(out, batch_first=False, padding_value=PAD)
    
    # <bos> 010010101 <eos>
    return out

def sample_lm_vary(model, batch_size_i, max_len=None):
    # output: [ seq_len x batch ]
    
    # [ 1 x batch ]
    # <pad> idx = 4
    if not max_len:
        max_len=args.n*2+1
    out = [(torch.ones(1)*3).cuda().long()]*batch_size_i # contains sequences of variable lenght
    symb = start_symbol[:,:batch_size_i]
    hidden = model.init_hidden(batch_size_i)
    len_inp = torch.ones((batch_size_i), dtype=torch.int64)
    mask = torch.ones((1, batch_size_i, 1)).cuda()
    all_logits = torch.ones((1, batch_size_i, ninp)).cuda()
    
    for i in range(max_len):
        # [1 x batch x ntok]
        logits, hidden = model(symb, hidden, len_inp, mask)[:2]
        probs = softm(logits)
        cat_dist = torch.distributions.Categorical(probs=probs)
        # [ 1 x batch ]
        symb = cat_dist.sample()
        flag = False
        for b in range(batch_size_i):
            if i==0 or (i>0 and out[b][-1] != 2): 
                out[b] = torch.cat((out[b], symb[:1,b]), dim=0)
                flag = True
        all_logits = torch.cat((all_logits, logits), dim=0)
        if not flag:
            break
        
    out = torch.nn.utils.rnn.pad_sequence(out, batch_first=False, padding_value=PAD)   
    # <bos> 010010101 <eos>
    return out, all_logits[1:out.size(0),:,:]

softm = nn.Softmax(dim=2)


# training
def upper_bound(params):
    out = torch.zeros((1)).cuda()
    for i in range(params.size(1)):
        curr = params[:1,i]
        if curr > 0:
            out = torch.cat((out, curr), dim=0)
    return torch.exp(out.sum(0))

def log_upper_bound(params):
    out = torch.zeros((1)).cuda()
    for i in range(params.size(1)):
        curr = params[:1,i]
        if curr > 0:
            out = torch.cat((out, curr), dim=0)
    return out.sum(0)


def sample_data_inp_targ(model, batch_size_i):
    # padded variable lengths sequences
    # [ seq_len x batch ]
    x = sample_lm(model, batch_size_i)
    len_inp = (x!= PAD).sum(0)
    len_inp, perm_idx = len_inp.sort(0, descending=True)
    len_inp = len_inp - 1
    x = x[:, perm_idx]
    inp = x[:-1,:]

    # [(n+1) x batch]
    targets = x[1:,:] 
    mask_tar = (targets != PAD).unsqueeze(2).float().cuda()   
    len_tar = (targets != PAD).sum(0)
    
    return x, inp, len_inp, targets, mask_tar


def sample_data_inp_targ_vary(model, batch_size_i, max_len=None):
    # padded variable lengths sequences
    # [ seq_len x batch ]
    if not max_len:
        max_len = args.n*2+1
    x, log_pi = sample_lm_vary(model, batch_size_i, max_len)
    len_inp = (x!= PAD).sum(0)
    len_inp, perm_idx = len_inp.sort(0, descending=True)
    len_inp = len_inp - 1
    x = x[:, perm_idx]
    inp = x[:-1,:]

    # [(n+1) x batch]
    targets = x[1:,:] 
    mask_tar = (targets != PAD).unsqueeze(2).float().cuda()   
    len_tar = (targets != PAD).sum(0)
    
    return x, log_pi, inp, len_inp, targets, mask_tar


def get_length_mask(x):
    len_inp = (x!= PAD).sum(0)
    len_inp, perm_idx = len_inp.sort(0, descending=True)
    len_inp = len_inp - 1
    x = x[:, perm_idx]
    inp = x[:-1,:]

    # [(n+1) x batch]
    targets = x[1:,:] 
    mask_tar = (targets != PAD).unsqueeze(2).float().cuda()
    len_tar = (targets != PAD).sum(0)
    
    return len_tar, mask_tar, inp, targets.contiguous()


def rejection_sampling(model, ce_criterion, motifs, feat, ro_stats):
    # q(x)=r(x), Q(x)>=P_lambda(x) for any x 
    # sample from LM: x ~ q(x)
    # accept with probability ro = P_lambda(x)/Q(x)

    nfeat = sum([sum([int(e!='0') for e in el]) for el in feat])
    samples = torch.ones((1, nfeat)).cuda()
    batch_size_i = 2*batch_size
    
    #accpt_samples = torch.ones((args.n+2, 1)).cuda().long()
    
    acceptance_rate, total_samples, accepted = ro_stats
    
    while samples.size(0) <= nsamples:            
        x, log_pi, inp, len_inp, targets, mask_tar = sample_data_inp_targ_vary(model, batch_size_i)

        hidden = model.init_hidden(batch_size_i)
        # log_lin  [ batch x 1 ]        
        r_output, _, log_lin = model(inp, hidden, len_inp, mask_tar) # outpt [seq_len ,batch, ntok]
        # [ batch x 1 ]
        #log_r = get_log_r(r_output, targets, log_lin, mask_tar, ce_criterion)
        #P_lambda = torch.exp(log_r + log_lin)

        # upper boundary: P_lambda <= q(x)*exp(max(lambda * feat))
        log_beta = log_upper_bound(model.lin_lambda.weight)
        ro = torch.exp(log_lin - log_beta)[:,0].cpu()
        
        acceptance_rate = (total_samples*acceptance_rate + ro.sum())/(total_samples+ro.size(0))
        indicator = torch.rand((ro.size(0))) <= ro
        total_samples += ro.size(0)
        accepted = accepted + indicator.sum().float()
        all_feats =  get_features(x, motifs, feat)

        for i in range(indicator.size(0)):
            if indicator[i]:
                feat_x = all_feats[i:i+1]
                samples = torch.cat((samples, feat_x), dim=0)
                #accpt_samples = torch.cat((accpt_samples, x[:,i:i+1]), dim=1)
                
    # samples [ nsamples x nfeat ]
    return samples[1:nsamples+1,:].mean(0), [acceptance_rate, total_samples, accepted]


# keep samples for fixed theta
def get_samples_rs(model, x, inp, len_inp, mask_tar, acceptance_rate, total_samples, motifs, feat, accepted, batch_size_i, samples):
    
    all_feats =  get_features(x, motifs, feat)      
    log_lin = model.lin_lambda(all_feats)
    # [ batch x 1 ]
    #log_r = get_log_r(r_output, targets, log_lin, mask_tar, ce_criterion)
    #P_lambda = torch.exp(log_r + log_lin)

    # upper boundary: P_lambda <= q(x)*exp(max(lambda * feat))
    log_beta = log_upper_bound(model.lin_lambda.weight)
    ro = torch.exp(log_lin - log_beta)[:,0].cpu()
    
    acceptance_rate = (total_samples*acceptance_rate + ro.sum())/(total_samples+ro.size(0))
    indicator = torch.rand((ro.size(0))) <= ro
    total_samples += ro.size(0)
    accepted = accepted + indicator.sum().float()

    for i in range(indicator.size(0)):
        if indicator[i]:
            feat_x = all_feats[i:i+1]
            samples = torch.cat((samples, feat_x), dim=0)
            #accpt_samples = torch.cat((accpt_samples, x[:,i:i+1]), dim=1)
    return samples, acceptance_rate, total_samples, accepted


def cat_variable_length(a, b):
    seq_len = max(a.size()[0], b.size()[0])
    if a.size()[0] < seq_len:
        a = torch.cat((a, torch.ones((seq_len-a.size()[0], a.size(1))).long().cuda()*PAD), dim=0)
    if b.size()[0] < seq_len:
        b = torch.cat((b, torch.ones((seq_len-b.size()[0], b.size(1))).long().cuda()*PAD), dim=0)

    return torch.cat((a,b), dim=1)

def cyclic_rejection_sampling(model, ce_criterion, motifs, feat, ro_stats, am_samples):
    # q(x)=r(x), Q(x)>=P_lambda(x) for any x 
    # sample from LM: x ~ q(x)
    # accept with probability ro = P_lambda(x)/Q(x)

    nfeat = sum([sum([int(e!='0') for e in el]) for el in feat])
    samples = torch.ones((1, nfeat)).cuda()
    batch_size_i = 2*batch_size
    
    #accpt_samples = torch.ones((args.n+2, 1)).cuda().long()
    
    acceptance_rate, total_samples, accepted = ro_stats

    len_inp, mask_tar, inp, targets = get_length_mask(am_samples)
    if am_samples.size(1) != 0:
        samples, acceptance_rate, total_samples, accepted = get_samples_rs(model, am_samples, inp, 
                len_inp, mask_tar, acceptance_rate, total_samples, motifs, feat, accepted, batch_size_i, samples)

    while samples.size(0) <= nsamples:            
        x, log_pi, inp, len_inp, targets, mask_tar = sample_data_inp_targ_vary(model, batch_size_i)

        samples, acceptance_rate, total_samples, accepted = get_samples_rs(model, x, inp, 
            len_inp, mask_tar, acceptance_rate, total_samples, motifs, feat, accepted, batch_size_i, samples)
    
        am_samples = cat_variable_length(am_samples, x)

    # samples [ nsamples x nfeat ]
    return samples[1:nsamples+1,:].mean(0), [acceptance_rate, total_samples, accepted], am_samples


def sample_data_inp_targ_snis(model, batch_size_i, source_data):
    # padded variable lengths sequences
    # [ seq_len x batch ]
    x_r, _ = sample_lm_vary(model, batch_size_i)

    if args.train == 'snis_mix':
        d = source_data.size(1)
        # [batch x |D|] empirical dsitribution    
        probs = torch.ones((batch_size_i, d))*(1.0/d)
        cat_dist = torch.distributions.Categorical(probs=probs)
        # [ batch ]
        d_idx = cat_dist.sample()
        # [ seq_len x batch x 1 ]
        x_d = source_data[:, d_idx]
        r_or_d = (torch.rand((batch_size_i))>0.5).cuda()
        seq_len = max(x_r.size()[0], x_d.size()[0])
        if x_r.size()[0] < seq_len:
            x_r = torch.cat((x_r, torch.ones((seq_len-x_r.size()[0], batch_size_i)).long().cuda()*PAD), dim=0)
        if x_d.size()[0] < seq_len:
            x_d = torch.cat((x_d, torch.ones((seq_len-x_d.size()[0], batch_size_i)).long().cuda()*PAD), dim=0)
        x = torch.where(r_or_d, x_r, x_d)

    elif args.train == 'snis_r':
        r_or_d = (torch.ones((batch_size_i))).cuda()
        x = x_r
    
    len_inp = (x!= PAD).sum(0)
    len_inp, perm_idx = len_inp.sort(0, descending=True)
    # adjust due to the fact of sampling in mixture
    max_len = torch.max(len_inp)
    x = x[:max_len,:]

    len_inp = len_inp - 1
    x = x[:, perm_idx]    
    inp = x[:-1,:]

    # [(n+1) x batch]
    targets = x[1:,:] 
    mask_tar = (targets != PAD).unsqueeze(2).float().cuda()   
    len_tar = (targets != PAD).sum(0)
    
    return x, inp, len_inp, targets, mask_tar, r_or_d


def snis(model, ce_criterion, motifs, feat, source_data, hash_source, total_feat, total_w):
    # q(x)=0.5r(x) + 0.5D(x), D(x) empirical distribution 
    # sample from LM: x ~ q(x)
    # weighted expectation w.r.t w_i = P_lambda(x_i)/q(x_i)
    # 

    nfeat = sum([sum([int(e!='0') for e in el]) for el in feat])
    batch_size_i = 2*batch_size
    
    x, inp, len_inp, targets, mask_tar, r_or_d = sample_data_inp_targ_snis(model, batch_size_i, source_data)
    # [batch x nfeat]
    all_feats = get_features(x, motifs, feat)
    hidden = model.init_hidden(batch_size_i)
    # log_lin  [ batch x 1 ]        
    r_output, _, log_lin = model(inp, hidden, len_inp, mask_tar) # outpt [seq_len ,batch, ntok]
    # [ batch x 1 ]
    log_r = get_log_r(r_output, targets, mask_tar, ce_criterion).sum(0)
    # P_lambda = torch.exp(log_r + log_lin)
    
    d = source_data.size(1)   
    probs = (torch.ones((batch_size_i))*(1.0/d)).cuda()
    if args.train == 'snis_mix':
        for b in range(x.size(1)):
            if r_or_d[b] == 0: continue
            x_i = ''.join([str(el) for el in x[:,b].cpu().numpy()])
            if hash(x_i) in hash_source and x_i in hash_source[hash(x_i)]:
                r_or_d[b] = 0 

        q = 0.5*torch.exp(log_r) + 0.5*torch.where(r_or_d, torch.zeros(probs.size()).cuda(), probs).unsqueeze(1)
        w = torch.exp(log_r + log_lin - torch.log(q))

        if total_feat[:,:].size(0) != 0:
            all_feats = torch.cat((total_feat, all_feats), dim=0)
            w = torch.cat((total_w, w), dim=0)

    elif args.train == 'snis_r':
        # q = torch.exp(log_r)
        if total_feat[:,:].size(0) != 0:
            all_feats = torch.cat((total_feat, all_feats), dim=0)
            log_lin = model.lin_lambda(all_feats)
        # log_r + log_lin - log_r
        w = torch.exp(log_lin)
    
    mean_feats = torch.mul(all_feats, w).sum(0)/w.sum()
                
    # samples [ nsamples x nfeat ]
    return mean_feats.detach(), w, all_feats


# training

def evaluate(model, criterion, source_data):
    model.eval()
    total_loss = 0
    batches_id = list(range(0, source_data.size(1), batch_size)) 
    for i, batch in enumerate(batches_id):
        len_tar, mask_tar, data, target = get_length_mask(source_data[:,batch:batch+batch_size])
        batch_size_i = mask_tar.size()[1]
        target = torch.mul(target.float(), mask_tar[:,:,0]).long()
        hidden = model.init_hidden(batch_size_i)
        output, hidden = model(data, hidden, 0, mask_tar)
        output_flat = output.view(-1, ninp)
        # [(n+1) x batch x 1]
        loss = criterion(output.view(-1, ninp), target.view(-1)).view(mask_tar.size())
        #if i == 0:
        #    print(output[5:15,45].squeeze().data.cpu().numpy(), 
        #          data[5:15,45].squeeze().data.cpu().numpy(), loss[5:15,45].squeeze().data.cpu().numpy())
        loss = torch.div(torch.mul(loss, mask_tar).sum(0).squeeze(), len_tar.float()).mean()
        #assert loss.size() == len_tar.float().unsqueeze(1).size()
        total_loss +=  loss.data.float()
    return total_loss / len(batches_id)

def evaluate_ce_pl_ds(model, ce_criterion, source_data, z_estim=None):
    # evaluate cross entropy on the whole dataset
    model.eval()
    total_loss = 0
    likelih = torch.tensor([[1.0/source_data.size(1)]]*batch_size).cuda()
    
    batches_id = list(range(0, source_data.size(1), batch_size)) 
    if not z_estim:
        z_estim = estimate_partition_mc(model, ce_criterion)
    for i, batch in enumerate(batches_id):
        #data, target = get_batch_fsz(source_data, batch)
        len_tar, mask_tar, data, target = get_length_mask(source_data[:,batch:batch+batch_size])
        batch_size_i = mask_tar.size()[1]
        hidden = model.init_hidden(batch_size_i)
        r_output, hidden, log_lin = model(data, hidden, len_tar, mask_tar)
        
        # [ batch x 1 ]
        log_r = get_log_r(r_output, target, mask_tar, ce_criterion).sum(0)
        P_lambda = torch.exp(log_r + log_lin)
        
        p_lambda = P_lambda/z_estim
        
        ce_loss = (-1*torch.mul(likelih, torch.log(p_lambda))).sum()
        
        total_loss +=  ce_loss.data.float()
    return total_loss  

def evaluate_ce_r(model, ce_criterion, source_data):
    model.eval()
    total_loss = 0

    batches_id = list(range(0, source_data.size(1), batch_size)) 
    for i, batch in enumerate(batches_id):
        #data, targets = get_batch(source_data, batch)
        len_tar, mask_tar, data, targets = get_length_mask(source_data[:,batch:batch+batch_size])
        batch_size_i = data.size()[1]
        hidden = model.init_hidden(batch_size_i)
        
        r_output, hidden, log_lin = model(data, hidden, len_tar, mask_tar)
        output_flat = r_output.view(-1, ninp)
        total_loss +=  ce_criterion(output_flat, targets.view(-1)).mean().data.float()

    return total_loss / len(batches_id)


def to_one_hot(y, n_dims=None):
    """ Take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else ninp
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1).cuda()
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def get_log_r(r_output, ce_target, mask_tar, ce_criterion):
    # mask PAD symbol to keep short input vocabulary
    ce_target = torch.mul(ce_target.float(), mask_tar[:,:,0]).long()
    # [(n+1) x batch x 1]
    #log_r_seq = torch.mul((-1*ce_criterion(r_output.view(-1, ninp), ce_target.view(-1))).view(mask_tar.size()), mask_tar)
    r_output = torch.nn.functional.log_softmax(r_output, dim=2)
    log_r_seq = torch.sum(r_output.view(-1, ninp) * to_one_hot(ce_target.view(-1)), dim = 1)
    log_r_seq = torch.mul(log_r_seq.view(mask_tar.size()), mask_tar)
    #log_r = log_r_seq.sum(0)  # [ batch x 1 ]
    return log_r_seq

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def estimate_partition_mc(model, ce_criterion):
    # Z_lambda = E_{x~q(.)}[P_lambda(x)/q(x)] = E_{x~q(.)}[exp(lambda^T feat(x))]
    # sample from q(x) = r(x), use IS to compute expectation w.r.t. p_lambda distribution
    # compute expectation using MC samples    
    batch_size_i = min(batch_size*50, 6500)
    x, log_pi, inp, len_inp, target, mask_tar = sample_data_inp_targ_vary(model, batch_size_i)
    
    hidden = model.init_hidden(batch_size_i)
    r_output, _, log_lin = model(inp, hidden, len_inp, mask_tar) # outpt [seq_len ,batch, ntok]
    # [ batch x 1 ]
    z_samples = torch.exp(log_lin)
    
    return z_samples.mean() 
         

def repackage_hidden(h):
    """detach vars from their history."""
    return tuple(Variable(h[i].data) for i in range(len(h)))

def train_r(model, criterion, epoch, source_data, lr, optimizer):
    
    total_loss = 0.
    start_time = time.time()
    
    #criterion2 = nn.CrossEntropyLoss()
    print(batch_size)
    batches_id = list(range(0, source_data.size(1), batch_size))
    shuffle(batches_id)

    all_idx = list(range(source_data.size(1)))
    shuffle(all_idx)
    source_data = source_data[:,all_idx]
    
    for i, batch in enumerate(batches_id):

        loss = single_update_r(model, source_data[:,batch:batch+batch_size], optimizer, lr, criterion)

        total_loss += loss.data.float()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| iter {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.5f}'.format(
                      epoch, i, len(batches_id),lr, get_lr(optimizer), 
                      elapsed * 1000 / log_interval, cur_loss, math.exp(min(cur_loss, 20))))
            total_loss = 0
            start_time = time.time()

def single_update_r(model, data, optimizer, lr, criterion):
    model.train()

    len_tar, mask_tar, data, target = get_length_mask(data)
    target = torch.mul(target.float(), mask_tar[:,:,0]).long()
    batch_size_i = mask_tar.size()[1]
    hidden = model.init_hidden(batch_size_i)
    model.zero_grad()
    optimizer.zero_grad()
    output, hidden = model(data, hidden, 0, mask_tar) # outpt [seq_len ,batch, ntok]
    
    loss = criterion(output.view(-1, ninp), target.view(-1)).view(mask_tar.size())
    loss = torch.div(torch.mul(loss, mask_tar).sum(0).squeeze(), len_tar.float()).mean()
    loss.backward()
    # to prevent the exploding gradient problem
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
    
    optimizer.step()

    return loss



def cyclic_train_lambda(model, ce_criterion, epoch, source_data, lr, 
                 motifs, feat, ro_stats, optimizer, target_feat, writer, hash_source, am_samples):
    model.train()
    total_loss = 0.
    start_time = time.time()
    
    batches_id = list(range(0, source_data.size(1)))
    shuffle(batches_id)
    source_data = source_data[:,batches_id]
    
    batches_id = list(range(0, source_data.size(1), batch_size))
    shuffle(batches_id)

    all_idx = list(range(source_data.size(1)))
    shuffle(all_idx)
    source_data = source_data[:,all_idx]
    
    nfeat = sum([sum([int(e!='0') for e in el]) for el in feat])
    all_mean_feat = torch.zeros(nfeat).cuda()
    total_feat = torch.zeros(0, nfeat).cuda()
    total_w = torch.zeros(0, 1).cuda()
    
    for i, batch in enumerate(batches_id):
        #data, target = get_batch_fsz(source_data, batch)
        len_tar, mask_tar, data, target = get_length_mask(source_data[:,batch:batch+batch_size])
        batch_size_i = data.size()[1]
        hidden = model.init_hidden(batch_size_i)
        model.zero_grad()
        r_output, hidden, log_lin = model(data, hidden, len_tar, mask_tar) # outpt [seq_len ,batch, ntok]
        
        if args.train == 'rs':
            mean_feat, ro_stats, am_samples = cyclic_rejection_sampling(model, ce_criterion, motifs, feat,ro_stats, am_samples)
            am_samples = am_samples[:,-50000:]
        elif 'snis' in args.train:
            # [batch x nfeat]
            mean_feat, curr_w, curr_feat = snis(model, ce_criterion, motifs, feat, source_data, hash_source, total_feat, total_w)
            total_feat = torch.cat((total_feat, curr_feat), dim=0)
            total_w = torch.cat((total_w, curr_w), dim=0)
            total_feat = total_feat[-10000:,:]
            total_w = total_w[-10000:,:]
        
        all_mean_feat += mean_feat
            
        #target_feat = get_features(target,motifs, feat)
        grads = (target_feat - mean_feat).unsqueeze(0)
        torch.nn.utils.clip_grad_norm_(grads, clip)
        
        #model.lin_lambda.weight.data.add_(lr, grads.data)
        model.lin_lambda.weight.grad = -grads
            
        for n, p in model.named_parameters():
            if  'lin_lambda.weight' in n: 
                p.data.add_(-lr, p.grad.data)
                       
        # [ batch x 1 ]
        log_r = get_log_r(r_output, target, mask_tar, ce_criterion).sum(0)
        P_lambda = torch.exp(log_r + log_lin)
        #z_estim = estimate_partition_mc(model, ce_criterion)
        #p_lambda = P_lambda/z_estim
        
        #print('r_output\n', r_output.size(), z_estim, target.view(-1).size(), target.size())
        #writer.add_histogram('data/lambdas', model.lin_lambda.weight.clone().cpu().data.numpy(), 
        #                  epoch*(len(batches_id))+i)
        
        total_loss += P_lambda.mean().data.float() #l1_loss.data.float()
        if i % log_interval == 0:# and i > 0:
            cur_loss = total_loss / (log_interval)
            elapsed = time.time() - start_time
            print('grads', grads.data.cpu().numpy())
            print('mean_feat', mean_feat.data.cpu().numpy(),'taget_feat', target_feat.data.cpu().numpy())
            print('lambda', model.lin_lambda.weight.data.squeeze().cpu().numpy(), model.lin_lambda.bias.data.squeeze().cpu().numpy())
            print('| iter {:3d} | {:5d}/{:5d} batches | lr {} | ms/batch {:5.10f} | '
                  'P_lambda {} '.format(epoch, i, len(batches_id), lr,
                      elapsed * 1000 / log_interval, cur_loss))
            
            total_loss = 0
            start_time = time.time()
            
            #x, inp, len_inp, targets, mask_tar = sample_data_inp_targ(model, 10)
    
    return ro_stats, all_mean_feat/len(batches_id), am_samples

#TODO: snis_mix fix accumulation of weights


def cyclic_distill_rejection_sampling(model, ce_criterion, motifs, feat, ro_stats, ds_size):
    # q(x)=r(x), Q(x)>=P_lambda(x) for any x 
    # sample from LM: x ~ q(x)
    # accept with probability ro = P_lambda(x)/Q(x)

    nfeat = sum([sum([int(e!='0') for e in el]) for el in feat])
    samples = [(torch.ones(1)*PAD).cuda().long()]*ds_size
    batch_size_i = 1024
    
    acceptance_rate, total_samples, accepted = ro_stats
    count = 0

    while count < ds_size:            
        #x, inp, len_inp, targets, mask_tar = sample_data_inp_targ(model, batch_size_i)
        x, log_pi, inp, len_inp, target, mask_tar = sample_data_inp_targ_vary(model, batch_size_i)

        #hidden = model.init_hidden(batch_size_i)
        # log_lin  [ batch x 1 ]        
        #r_output, _, log_lin = model(inp, hidden, len_inp, mask_tar) # r_outpt [seq_len ,batch, ntok]
        # [ batch x 1 ]
        #log_r = get_log_r(r_output, targets, log_lin, mask_tar, ce_criterion)
        #P_lambda = torch.exp(log_r + log_lin)

        all_feats =  get_features(x, motifs, feat)      
        log_lin = model.lin_lambda(all_feats)

        # upper boundary: P_lambda(x) <= q(x)*exp(max(lambda * feat))
        log_beta = log_upper_bound(model.lin_lambda.weight)
        ro = torch.exp(log_lin - log_beta)[:,0].cpu()
        
        
        acceptance_rate = (total_samples*acceptance_rate + ro.sum())/(total_samples+ro.size(0))
        indicator = torch.rand((ro.size(0))) <= ro
        total_samples += ro.size(0)
        accepted = accepted + indicator.sum().float()

        for i in range(indicator.size(0)):
            if indicator[i]:
                if count >= ds_size:
                    break

                samples[count] = torch.cat((samples[count], x[:,i]), dim=0)
                count += 1

        if count % 25 == 0:
            print('ro', acceptance_rate.data.cpu().numpy(), 'rate', (accepted)/total_samples)
 

    samples_cat = torch.nn.utils.rnn.pad_sequence(samples, batch_first=False, padding_value=PAD)
                   
    # samples [ seq_len x ds_size ]
    return samples_cat[1:,:ds_size], [acceptance_rate, total_samples, accepted]


def sample_from_rnn(model, n, motifs):
    batch_size = 2000
    x, log_pi, inp, len_inp, action, mask_tar = sample_data_inp_targ_vary(model, 
                                                    batch_size, max_len=500)
    avg_len = np.round(len_inp.float().mean().data.cpu().numpy(),decimals=1)
    print('avg len ', avg_len)
    x = x.data.cpu().numpy()
    count = 0     
    for i in range(x.shape[1]):
            res = ''.join([str(x[j,i]) for j in range(x.shape[0])])
            curr_count = 0
            for motif in motifs:
                if motif in res:
                    curr_count += 1
            count += min(1, curr_count)

    print('%d motifs in total %d' % (count, x.shape[1]))
    motif_freq = (1.0*count)/x.shape[1]
    return motif_freq, avg_len



def r_plambda_distill_pitheta():

    Epoch_start_time = time.time() 


    global batch_size
    n = args.n
    # ----------------------------------------------- train r on original datset ------------------------------------------
    batch_size = 500
    batch_size = min(args.ds_size, batch_size)
    lr = 0.001
    log_dir = '/tmp-network/user/tparshak/exp_gams/runs/chpt_%s'%(timestamp)
    os.mkdir(log_dir)
    epochs = args.epochs
    train_size = size = args.ds_size  
    feat = [args.feat]
    motifs = all_motifs[args.n].split('.')
    print('motif %s'%motifs[0])

    # (seq_len, nbatch)
    test_size = 5000
    valid_size = min(max(batch_size, int(0.25*train_size)), 2000)

    train_data_D = train_data = load_data_motif(n, train_size, all_motifs[args.n], 'train')
    valid_data_V = valid_data = load_data_motif(n, valid_size, all_motifs[args.n], 'valid')
    test_data  = load_data_motif(n, test_size, all_motifs[args.n], 'test')
    train_feat = get_features(train_data, motifs, feat).mean(0)
    
    print('orig train ds feat = ', train_feat.cpu().numpy())
    print('orig valid ds feat = ', get_features(valid_data,motifs, feat).mean(0).cpu().numpy())
    print('orig test ds feat = ', get_features(test_data,motifs, feat).mean(0).cpu().numpy())
        
    best_val_loss = None
    counter = 0
    patience = 10

    model_r = RNNModel(ntoken, ninp, nhid, nlayers, dropout)
    model_r.cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    optimizer_r = optim.Adam(model_r.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_r(model_r, criterion, epoch, train_data, lr, optimizer_r)
        val_loss = evaluate(model_r, criterion, valid_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
              'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(min(val_loss, 20))))
        print('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(log_dir,'chpt_%s_r.pt'%(timestamp)), 'wb') as f:
                torch.save(model_r, f)
            best_val_loss = val_loss
            conter = 0
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if best_val_loss:
                counter += 1
                if counter >= patience:
                    break
    del model_r 
    model_r = torch.load(os.path.join(log_dir,'chpt_%s_r.pt'%(timestamp)))
    test_loss = evaluate(model_r, criterion, test_data)
    train_loss = evaluate(model_r, criterion, train_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {} | '
          'test ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                     test_loss, math.exp(min(test_loss, 20))))
    print('-' * 89)
    
    entp = entp_motifs[n]
    print('\nTheoretical entp = {} for n = {:2d}  \n'.format(entp,  n))

    test_ce_r = test_loss.float().cpu().numpy().item(0)

    # ------------------------------------- train P_lambda -----------------------------------------
    batch_size = 500
    batch_size = min(batch_size, args.ds_size)

    lr0 = lr = 10.#0.001
    n = args.n
    train_size = size = args.ds_size  
    feat = [args.feat]
    motifs = all_motifs[args.n].split('.')
    print('motifs ', motifs)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_loss = None
    counter = 0

    model_plambda = GAMModel(ntoken, ninp, nhid, nlayers, feat, motifs, dropout)
    model_plambda.cuda()
    ce_criterion = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    optimizer_pl = optim.Adam(model_plambda.parameters(), lr=lr)

    hash_train = {}
    if args.train == 'snis_mix':
        for b in range(train_data.size(1)):
            x_i = ''.join([str(el) for el in train_data[:,b].cpu().numpy()])
            if hash(x_i) in hash_train:
                hash_train[hash(x_i)] += [x_i]
            else:
                hash_train[hash(x_i)] = [x_i] 

    if args.theta_fixed:
        model_dict = model_plambda.state_dict()
        pretrained_model = model_r
        pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() 
                                if (k in model_dict) and (v.size() == model_dict[k].size())}
        model_dict.update(pretrained_dict)
        model_plambda.load_state_dict(model_dict)
        
        test_loss_r = evaluate_ce_r(model_plambda, ce_criterion, test_data)
        train_loss_r = evaluate_ce_r(model_plambda, ce_criterion, train_data)

        print("test_r {}, train_r {}".
            format(test_loss_r.data.float(), train_loss_r.data.float(),))
        
        print('\nTheoretical entp = {:5.4f} for n = {:2d}  \n'.format(entp, n))
        
    acceptance_rate = torch.zeros((1)).cuda()
    total_samples = 0
    accepted = 0
    ro_stats = [acceptance_rate, total_samples, accepted]                

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        if args.theta_fixed:
            ro_stats, mean_feat = train_lambda(model_plambda, ce_criterion, epoch, 
                                    train_data, lr, motifs, feat, ro_stats, optimizer_pl, train_feat, writer, hash_train)
        #val_loss = evaluate_ce_pl_ds(model_plambda, ce_criterion, valid_data)/(n+1)
        l1_feat = torch.abs(train_feat - mean_feat).sum()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | l1_feat {} '.
              format(epoch, (time.time() - epoch_start_time), l1_feat))
        print('lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy(), model_plambda.lin_lambda.bias.data.squeeze().cpu().numpy())
        if args.train == 'rs':
            acceptance_rate, total_samples, accepted = ro_stats
            print('ro', acceptance_rate.data.cpu().numpy(), 'rate', (accepted)/total_samples) 
            #writer.add_scalar('data/accpt_rate', (accepted)/total_samples, epoch)
        print('mean_feat', mean_feat.data.squeeze().cpu().numpy())
        print('-' * 89)
        
        #writer.add_scalar('data/lr', lr, epoch)
        #writer.add_scalar('data/val_ds_ce', val_loss, epoch)
        #writer.add_histogram('data/mean_feat', mean_feat, epoch)        
        #writer.add_scalar('data/l1_feat', l1_feat, epoch)

        lr = lr0/(1 + epoch)
                        
        if not best_val_loss or l1_feat < best_val_loss:
            with open(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)), 'wb') as f:
                torch.save(model_plambda, f)
            best_val_loss = l1_feat #val_loss
            conter = 0
        else:
            if best_val_loss:
                counter += 1
                if counter >= patience:
                    break
    del model_plambda 
    model_plambda = torch.load(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)))
    writer.close()
    print('-' * 89)

    plambda_time = time.time() - start_time
    
    # hybrid model
    z_estim = estimate_partition_mc(model_plambda, ce_criterion)
    test_loss = evaluate_ce_pl_ds(model_plambda, ce_criterion, test_data, z_estim)/(n+1)
    train_loss = evaluate_ce_pl_ds(model_plambda, ce_criterion, train_data, z_estim)/(n+1)
    
    # autoregressive model
    test_loss_r = evaluate_ce_r(model_plambda, ce_criterion,  test_data)
    train_loss_r = evaluate_ce_r(model_plambda, ce_criterion, train_data)

    print('\nTheoretical entp = {:5.4f} for n = {:2d}  \n'.format(entp, n))
    print("test {}, train {}, n {}, ds_size {}, motif {}".
        format(test_loss.data.float(), train_loss.data.float(), n, size, motifs))
    print("test_r {}, train_r {}".
        format(test_loss_r.data.float(), train_loss_r.data.float()))
    print('-' * 89)
    print('lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy())

    #del model
    os.remove(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)))
    
    test_ce_pl,train_l1_pl,theor_ent,tstamp,lambd =  [test_loss.data.float().cpu().numpy().item(0), best_val_loss.data.float().cpu().numpy().item(0), entp, timestamp, 
                    str(list(model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy()))]

    # --------------------------------------- distill from P_lambda -----------------------------------------------------
    batch_size = min(args.distill_size, 1024)
    train_size = size = args.distill_size  

    # (seq_len, nbatch)
    valid_size = min(max(batch_size, int(0.2*train_size)), 2000)

    ce_criterion = nn.CrossEntropyLoss(reduction='none')

    # GAMModel
    print('lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy(), model_plambda.lin_lambda.bias.data.squeeze().cpu().numpy())
        
    acceptance_rate = torch.zeros((1)).cuda()
    total_samples = 0
    accepted = 0
        
    ro_stats = [acceptance_rate, total_samples, accepted]                

    train_data, ro_stats = distill_rejection_sampling(model_plambda, ce_criterion, motifs, feat, ro_stats, size)
    valid_data, ro_stats = distill_rejection_sampling(model_plambda, ce_criterion, motifs, feat, ro_stats, valid_size)

    print('train_data', train_data.size(), 'val', valid_data.size())
    
    

    
    print('-' * 89)
    acceptance_rate, total_samples, accepted = ro_stats
    print('ro', acceptance_rate.data.cpu().numpy(), 'rate', (accepted)/total_samples, 'num', 
                  total_samples, 'accpt', accepted)


    # ----------------------------------------- train pi_theta on distilled ds -----------------------------------------
    batch_size = min(args.distill_size, 500)

    lr = 0.001
    best_val_loss = None
    counter = 0

    model_pitheta = RNNModel(ntoken, ninp, nhid, nlayers, dropout)
    model_pitheta.cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    optimizer_pi = optim.Adam(model_pitheta.parameters(), lr=lr)

    train_data = cat_variable_length(train_data, train_data_D)
    valid_data = cat_variable_length(valid_data, valid_data_V)
    print('distilled_D', train_data.size())
    print('distilled_V', valid_data.size())

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        
        train_r(model_pitheta, criterion, epoch, train_data, lr, optimizer_pi)
        val_loss = evaluate(model_pitheta, criterion, valid_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
              'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(min(val_loss, 20))))
        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)), 'wb') as f:
                torch.save(model_pitheta, f)
            best_val_loss = val_loss
            conter = 0
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if best_val_loss:
                counter += 1
                if counter >= patience:
                    break

    Final_duration = (time.time() - Epoch_start_time)/3600.

    print('train ds feat (P_lambda) = ', get_features(train_data, motifs, feat).mean(0).cpu().numpy())
    print('valid ds feat (P_lambda) = ', get_features(valid_data, motifs, feat).mean(0).cpu().numpy())
    print('test  ds feat (original) = ', get_features(test_data,motifs, feat).mean(0).cpu().numpy())

    train_feat_pl = get_features(train_data, motifs, feat)
    valid_feat_pl = get_features(valid_data, motifs, feat)
    print('plambda train ds feat = ', train_feat_pl.mean(0).cpu().numpy())
    print('plambda valid ds feat = ', valid_feat_pl.mean(0).cpu().numpy())
    print('orig test ds feat = ', get_features(test_data, motifs, feat).mean(0).cpu().numpy())
    
    mfeat_pl = ((train_feat_pl.sum(0) + valid_feat_pl.sum(0))/(train_feat_pl.size(0)+valid_feat_pl.size(0))).cpu().numpy()
    train_l1_pl = np.absolute(train_feat.data.float().cpu().numpy() - mfeat_pl).item(0)
    mfeat_pl = str(list(mfeat_pl))

    np.save(os.path.join(log_dir,'train_n%d_f%s_m%d.npy'%(args.n, feat[0], args.motif)), train_data.cpu().numpy())
    np.save(os.path.join(log_dir,'valid_n%d_f%s_m%d.npy'%(args.n, feat[0], args.motif)), valid_data.cpu().numpy())

    del model_pitheta 
    model_pitheta = torch.load(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)))
    test_loss = evaluate(model_pitheta, criterion, test_data)
    train_loss = evaluate(model_pitheta, criterion, train_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {} | '
          'test ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                     test_loss, math.exp(min(test_loss, 20))))
    print('-' * 89)
    
    test_ce_pi = test_loss.float().cpu().numpy().item(0)

    # ------------------------------------------ sample from pi_theta ------------------------------------

    motif_freq, avg_len = sample_from_rnn(model_pitheta, n, motifs)
    motif_freq_r, avg_len_r = sample_from_rnn(model_r, n, motifs)
    print('r avg_len', avg_len_r, 'r motif_freq', motif_freq_r)
    print('pi avg_len', avg_len, 'pi motif_freq', motif_freq)

    tstamp = tstamp+'_'+str(motif_freq)+'_'+str(avg_len) +'_'+str(motif_freq_r)+'_'+str(avg_len_r)

    return [test_ce_r,test_ce_pi,test_ce_pl,train_l1_pl,theor_ent,tstamp,lambd,mfeat_pl,Final_duration]



def cyclic_r_plambda_pitheta():

    Epoch_start_time = time.time()

    global batch_size
    n = args.n
    # ----------------------------------------------- train initial r on true dataset D ------------------------------------------
    batch_size = 500
    batch_size = min(args.ds_size, batch_size)
    lr = 0.001
    log_dir = '/tmp-network/user/tparshak/exp_gams/runs/chpt_%s'%(timestamp)
    os.mkdir(log_dir)
    epochs = args.epochs
    train_size = size = args.ds_size  
    feat = [args.feat]
    motifs = all_motifs[args.n].split('.')
    print('motif %s'%motifs[0])

    # (seq_len, nbatch)
    test_size = 5000
    valid_size = min(max(batch_size, int(0.25*train_size)), 2000)

    train_data_D = load_data_motif(n, train_size, all_motifs[args.n], 'train')
    valid_data_V = load_data_motif(n, valid_size, all_motifs[args.n], 'valid')
    test_data_T  = load_data_motif(n, test_size, all_motifs[args.n], 'test')
    train_feat = get_features(train_data_D, motifs, feat).mean(0)
    
    print('orig train ds feat = ', train_feat.cpu().numpy())
    print('orig valid ds feat = ', get_features(valid_data_V,motifs, feat).mean(0).cpu().numpy())
    print('orig test ds feat = ', get_features(test_data_T,motifs, feat).mean(0).cpu().numpy())
        
    best_val_loss = None
    counter = 0
    patience = 10

    model_r = RNNModel(ntoken, ninp, nhid, nlayers, dropout)
    model_r.cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    optimizer_r = optim.Adam(model_r.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_r(model_r, criterion, epoch, train_data_D, lr, optimizer_r)
        val_loss = evaluate(model_r, criterion, valid_data_V)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
              'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(min(val_loss, 20))))
        print('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(log_dir,'chpt_%s_r.pt'%(timestamp)), 'wb') as f:
                torch.save(model_r, f)
            best_val_loss = val_loss
            conter = 0
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if best_val_loss:
                counter += 1
                if counter >= patience:
                    break
    del model_r 
    model_r = torch.load(os.path.join(log_dir,'chpt_%s_r.pt'%(timestamp)))
    test_loss = evaluate(model_r, criterion, test_data_T)
    train_loss = evaluate(model_r, criterion, train_data_D)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {} | '
          'test ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                     test_loss, math.exp(min(test_loss, 20))))
    print('-' * 89)
    
    entp = entp_motifs[n]
    print('\nTheoretical entp = {} for n = {:2d}  \n'.format(entp,  n))

    test_ce_r = test_loss.float().cpu().numpy().item(0)

    # ------------------------------------- training cycle: P_lambda + pi_theta -----------------------------------------

    best_val_loss_pi_dist = best_val_loss

    batch_size = 500
    batch_size = min(batch_size, args.ds_size)

    updates_per_epoch = int((1.0*args.distill_size)/batch_size)

    total_loss_cycl = 0.
    lr0 = 10.#0.001
    lr = 0.001
    n = args.n
    train_size = size = args.ds_size  
    feat = [args.feat]
    motifs = all_motifs[args.n].split('.')
    print('motifs ', motifs)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_loss_pi = None
    counter_pi = 0
    Epochs = 50

    model_plambda = GAMModel(ntoken, ninp, nhid, nlayers, feat, motifs, dropout)
    model_plambda.cuda()
    ce_criterion = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    optimizer_pl = optim.Adam(model_plambda.parameters(), lr=lr)

    hash_train = {}
    if args.train == 'snis_mix':
        for b in range(train_data.size(1)):
            x_i = ''.join([str(el) for el in train_data[:,b].cpu().numpy()])
            if hash(x_i) in hash_train:
                hash_train[hash(x_i)] += [x_i]
            else:
                hash_train[hash(x_i)] = [x_i] 


    distilled_D = torch.zeros((args.n+2,0)).cuda().long()
    distilled_V = valid_data_V #torch.zeros((args.n+2,0)).cuda().long()
    distilled_V_size = min(max(batch_size, int(0.25*args.distill_size)), 2000)
    model_pitheta = model_r

    patience_distl = 5
    patience_dist_pi = 15
    counter_dist= 0
    flag_pi_distill = False

    criterion = nn.CrossEntropyLoss(reduction='none')
                
    #optimizer_pi = optim.Adam(model_pitheta.parameters(), lr=lr)
    flag = False

    best_distill_acceptance_rate = None

    for Epoch in range(Epochs): 
        print('----- epoch %d ------'%Epoch)

        if distilled_D.size(1) < args.distill_size:
            for b in range(updates_per_epoch):
                if not flag_pi_distill:
                    # ------------------------------------- train P_lambda on the true dataset D -----------------------------------------

                    lr_pl = lr0

                    best_val_loss = None
                    counter = 0
                    model_dict = model_plambda.state_dict()
                    pretrained_dict = {k: v for k, v in model_pitheta.state_dict().items() 
                                            if (k in model_dict) and (v.size() == model_dict[k].size())}
                    model_dict.update(pretrained_dict)
                    model_plambda.load_state_dict(model_dict)

                    model_plambda.lin_lambda.bias.data = model_plambda.lin_lambda.bias.data * 0
                    model_plambda.lin_lambda.weight.data = model_plambda.lin_lambda.weight.data * 0
                    
                    test_loss_r = evaluate_ce_r(model_plambda, ce_criterion, test_data_T)
                    train_loss_r = evaluate_ce_r(model_plambda, ce_criterion, train_data_D)

                    print("test_r {}, train_r {}".
                        format(test_loss_r.data.float(), train_loss_r.data.float(),))
                    
                    print('\nTheoretical entp = {:5.4f} for n = {:2d}  \n'.format(entp, n))
                    
                    acceptance_rate = torch.zeros((1)).cuda()
                    total_samples = 0
                    accepted = 0
                    ro_stats = [acceptance_rate, total_samples, accepted]  

                    # keep samples from the fixed theta
                    am_samples = torch.zeros((args.n+2,0)).long().cuda()              

                    for epoch in range(1, epochs+1):
                        epoch_start_time = time.time()
                        if args.theta_fixed:
                            ro_stats, mean_feat, am_samples = cyclic_train_lambda(model_plambda, ce_criterion, epoch, 
                                                    train_data_D, lr_pl, motifs, feat, ro_stats, optimizer_pl, train_feat, writer, hash_train, am_samples)
                        #val_loss = evaluate_ce_pl_ds(model_plambda, ce_criterion, valid_data)/(n+1)
                        l1_feat = torch.abs(train_feat - mean_feat).sum()
                        print('-' * 89)
                        print('| end of epoch {:3d} | time: {:5.2f}s | l1_feat {} '.
                              format(epoch, (time.time() - epoch_start_time), l1_feat))
                        print('lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy(), model_plambda.lin_lambda.bias.data.squeeze().cpu().numpy())
                        if args.train == 'rs':
                            acceptance_rate, total_samples, accepted = ro_stats
                            print('ro', acceptance_rate.data.cpu().numpy(), 'rate', (accepted)/total_samples) 
                            #writer.add_scalar('data/accpt_rate', (accepted)/total_samples, epoch)
                        print('mean_feat', mean_feat.data.squeeze().cpu().numpy())
                        print('-' * 89)
                        
                        lr_pl = lr0/(1 + epoch)
                                        
                        if not best_val_loss or l1_feat < best_val_loss:
                            with open(os.path.join(log_dir, 'chpt_%s_pl_i.pt'%(timestamp)), 'wb') as f:
                                torch.save(model_plambda, f)
                            best_val_loss = l1_feat #val_loss
                            counter = 0
                        else:
                            if best_val_loss:
                                counter += 1
                                if counter >= patience_distl:
                                    break
                    del model_plambda 
                    model_plambda = torch.load(os.path.join(log_dir, 'chpt_%s_pl_i.pt'%(timestamp)))
                    writer.close()
                    print('-' * 89)

                    del am_samples

                    plambda_time = time.time() - start_time

                    print('\nTheoretical entp = {:5.4f} for n = {:2d}  \n'.format(entp, n))
                    print('-' * 89)
                    print('lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy())

                    #del model
                    os.remove(os.path.join(log_dir, 'chpt_%s_pl_i.pt'%(timestamp)))
                    
                    test_ce_pl,train_l1_pl,theor_ent,tstamp,lambd =  [999, best_val_loss.data.float().cpu().numpy().item(0), entp, timestamp, 
                                    str(list(model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy()))]

                # --------------------------------------- distill from P_lambda -----------------------------------------------------
                
                if b == 0:
                    with open(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)), 'wb') as f:
                            torch.save(model_plambda, f)

                train_size = size = batch_size  

                # (seq_len, nbatch)
                valid_size = int((1.0*distilled_V_size)/((1.0*args.distill_size)/batch_size))
                print('orig test ds feat = ', get_features(test_data_T,motifs, feat).mean(0).cpu().numpy())

                ce_criterion = nn.CrossEntropyLoss(reduction='none')

                # GAMModel
                print('lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy(), model_plambda.lin_lambda.bias.data.squeeze().cpu().numpy())
                    
                acceptance_rate = torch.zeros((1)).cuda()
                total_samples = 0
                accepted = 0
                    
                ro_stats = [acceptance_rate, total_samples, accepted]                

                train_data, ro_stats = cyclic_distill_rejection_sampling(model_plambda, ce_criterion, motifs, feat, ro_stats, size)
                valid_data, ro_stats = cyclic_distill_rejection_sampling(model_plambda, ce_criterion, motifs, feat, ro_stats, valid_size)

                distilled_D = cat_variable_length(distilled_D, train_data)
                distilled_V = cat_variable_length(distilled_V, valid_data)

                print('-' * 89)
                acceptance_rate, total_samples, accepted = ro_stats
                print('ro', acceptance_rate.data.cpu().numpy(), 'rate', (accepted)/total_samples, 'num', 
                              total_samples, 'accpt', accepted)

                if not flag_pi_distill:
                    # ---------------------------------------- pi_theta update on one batch -------------------------------------
                    
                    #val_loss_pi_distl = evaluate(model_pitheta, criterion, valid_data_V)
                    distill_acceptance_rate = acceptance_rate.data.cpu().numpy().item(0)

                    print('-' * 89)
                    print('| distill acceptance_rate {} '.format(distill_acceptance_rate))
                    print('-' * 89)

                    if not best_distill_acceptance_rate or distill_acceptance_rate > best_distill_acceptance_rate:

                        with open(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)), 'wb') as f:
                            torch.save(model_plambda, f)
                        best_distill_acceptance_rate = distill_acceptance_rate 
                        counter_dist = 0

                    else:
                        if best_distill_acceptance_rate:
                            counter_dist += 1
                            if counter_dist >= patience_dist_pi:
                                flag_pi_distill = True
                                model_plambda = torch.load(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)))
                    

                    ce_loss = single_update_r(model_pitheta, train_data, optimizer_r, lr, criterion)
                    total_loss_cycl += ce_loss.data.float()


                    if b % log_interval == 0 and b > 0:
                        cur_loss = total_loss_cycl / log_interval
                        elapsed = time.time() - start_time
                        print('| iter {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                              'loss {:5.2f} | ppl {:8.5f}'.format(
                                  Epoch, b, updates_per_epoch, lr, 
                                  elapsed * 1000 / log_interval, cur_loss, math.exp(min(cur_loss, 20))))

                if distilled_D.size(1) >= args.distill_size:
                    break

                print('distilled_D size', distilled_D.size())

        else:
            # ----------------------------------------- train pi_theta on distilled ds -----------------------------------------
            if not flag:
                flag = True
                distilled_D = cat_variable_length(distilled_D, train_data_D)
                print('distilled_D', distilled_D.size())
                print('distilled_V', distilled_V.size())

                model_pitheta = RNNModel(ntoken, ninp, nhid, nlayers, dropout)
                model_pitheta.cuda()
                optimizer_pi = optim.Adam(model_pitheta.parameters(), lr=lr)


            epoch_start_time = time.time()
            train_r(model_pitheta, criterion, Epoch, distilled_D, lr, optimizer_pi)
            val_loss = evaluate(model_pitheta, criterion, distilled_V)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
                  'valid ppl {}'.format(Epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(min(val_loss, 20))))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss_pi or val_loss < best_val_loss_pi:
                with open(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)), 'wb') as f:
                    torch.save(model_pitheta, f)
                best_val_loss_pi = val_loss
                counter_pi = 0
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                if best_val_loss_pi:
                    counter_pi += 1
                    if counter_pi >= patience:
                        break

    # hybrid model
    z_estim = estimate_partition_mc(model_plambda, ce_criterion)
    test_ce_pl = (evaluate_ce_pl_ds(model_plambda, ce_criterion, test_data_T, z_estim)/(n+1)).data.float().cpu().numpy().item(0)

    Final_duration = (time.time() - Epoch_start_time)/3600.

    del model_pitheta 
    model_pitheta = torch.load(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)))
    test_loss = evaluate(model_pitheta, criterion, test_data_T)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {} | '
          'test ppl {}'.format(Epoch, (time.time() - epoch_start_time),
                                     test_loss, math.exp(min(test_loss, 20))))
    print('-' * 89)
    
    test_ce_pi = test_loss.float().cpu().numpy().item(0)


    print('train ds feat (P_lambda) = ', get_features(distilled_D, motifs, feat).mean(0).cpu().numpy())
    print('valid ds feat (P_lambda) = ', get_features(distilled_V, motifs, feat).mean(0).cpu().numpy())
    print('test  ds feat (original) = ', get_features(test_data_T,motifs, feat).mean(0).cpu().numpy())

    np.save(os.path.join(log_dir,'train_n%d_f%s_m%d.npy'%(args.n, feat[0], args.motif)), distilled_D.cpu().numpy())
    np.save(os.path.join(log_dir,'valid_n%d_f%s_m%d.npy'%(args.n, feat[0], args.motif)), distilled_V.cpu().numpy())

    train_feat_pl = get_features(distilled_D, motifs, feat)
    valid_feat_pl = get_features(distilled_V, motifs, feat)

    mfeat_pl = ((train_feat_pl.sum(0) + valid_feat_pl.sum(0))/(train_feat_pl.size(0)+valid_feat_pl.size(0))).cpu().numpy()
    train_l1_pl = np.absolute(train_feat.data.float().cpu().numpy() - mfeat_pl).item(0)
    mfeat_pl = str(list(mfeat_pl))

    # ------------------------------------------ sample from pi_theta and r ------------------------------------
    del model_r 
    model_r = torch.load(os.path.join(log_dir,'chpt_%s_r.pt'%(timestamp)))

    motif_freq, avg_len = sample_from_rnn(model_pitheta, n, motifs)
    motif_freq_r, avg_len_r = sample_from_rnn(model_r, n, motifs)
    print('r avg_len', avg_len_r, 'r motif_freq', motif_freq_r)
    print('pi avg_len', avg_len, 'pi motif_freq', motif_freq)

    tstamp = tstamp+'_'+str(motif_freq)+'_'+str(avg_len) +'_'+str(motif_freq_r)+'_'+str(avg_len_r)

    return [test_ce_r,test_ce_pi,test_ce_pl,train_l1_pl,theor_ent,tstamp,lambd,mfeat_pl,Final_duration]



def main():  
    if args.cyclic:
        # cyclic improvement of pi -> r
        print('CYCLIC MODE')
        return cyclic_r_plambda_pitheta()
    else:
        #distill in one cyclic iteration
        print('distill in one cyclic iteration')
        return r_plambda_distill_pitheta()

if __name__ == "__main__":
    info_p_lambda = main()
    if args.cyclic:
        args.train = args.train+'_c'
    print(tuple(info_p_lambda+[args.mtype+'.'+all_motifs[args.n],args.train,args.feat,args.n,args.ds_size,args.job]))
    if not args.test_run:
        with sqlite3.connect('/tmp-network/user/tparshak/r_plambda_distill_pitheta.db') as conn:
            # this will be executed once because of the "IF NOT EXISTS" clause
            conn.execute('CREATE TABLE IF NOT EXISTS results (test_ce_r REAL,test_ce_pi REAL,test_ce_pl REAL,train_l1_pl REAL,theor_ent REAL,tstamp TEXT,lambd TEXT,mfeat_pl TEXT,plambda_time REAL,motif TEXT,train_reg TEXT,feat TEXT,n INTEGER,ds_size INTEGER,job INTEGER)')
            conn.execute('INSERT INTO results (test_ce_r,test_ce_pi,test_ce_pl,train_l1_pl,theor_ent,tstamp,lambd,mfeat_pl,plambda_time,motif,train_reg,feat,n,ds_size,job) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                tuple(info_p_lambda+[args.mtype+'.'+all_motifs[args.n],args.train,args.feat,args.n,args.ds_size,args.job]))
