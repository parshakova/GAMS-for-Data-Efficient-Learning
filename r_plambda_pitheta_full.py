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
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--ds_size', type=int, default=1000, help='training set size')
parser.add_argument('--distill_size', type=int, default=20000, help='training set size')
parser.add_argument('--motif', type=int, default=2, help='=1= short motif, =4= long motif')
parser.add_argument('--nmotifs', type=int, default=1, help='number of motifs that define the process')
parser.add_argument('--mtype', type=str, default='m', help='m, mam, m1m2, mult')
parser.add_argument('--n', type=int, default=30, help='string size')
parser.add_argument('--p', type=float, default=0.5, help='probability of flipping a coin')
parser.add_argument('--print_softm', type=str, default='', help='train or print')
parser.add_argument('--job', type=int, default=0, help='slurm job id')

#parser.add_argument('--feat', type=str, default='111', help='features for motifs with -.- separator; 0 or 1 at i-th position adds 0 to motif')
#parser.add_argument('--feat', type=str, default='1101000', help='features for motifs with -.- separator; (motif, supermotif, submotif, 1st bit==0, 10101, 1001001, 00110011)')
parser.add_argument('--feat', type=str, default='1001111', help='features for motifs with -.- separator; (motif, supermotif, submotif__2, 1st bit==0, 10101_len_m, 1001001_le_m_2, 00110011_len_m__2)')
parser.add_argument('--train', type=str, default='rs', help='=rs= rejection sampling, =snis_mix= snis mixture, =snis_r= snis r')
parser.add_argument('--restore', type=str, default='', help='checkpoint to restore model from')
parser.add_argument('--theta_fixed', action='store_false', help='train theta with lambda (log-linear model) or only lambda')
parser.add_argument('--test_run', action='store_true', help='if False - testing run, do not store accuracies')
parser.add_argument('--train2', type=str, default='distill', help='=distill=, =pg=, =dpg=, =cyclic_1=, =cyclic_r=')
parser.add_argument('--optim', type=str, default='adam', help='=adam=, =manual_lr=')
parser.add_argument('--debug_opt', type=str, default='no_motif', help='=no_motif=, =fix_length=')
parser.add_argument('--logdir', type=str, default='/tmp-network/user/tparshak')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--expect_len', type=float, default=30, help='expected length of strings in PFSA')
# hype parameters
parser.add_argument('--rl_lr', type=float, default=0.01, help='reinforcement learning learning rate')
parser.add_argument('--rl_scale_iter', type=float, default=100, help='reinforcement learning scaled number of iterations in one epoch')
parser.add_argument('--rl_target_kl', type=float, default=0.01, help='early stopping in ppo')
parser.add_argument('--rl_clip_param', type=float, default=0.2, help='in ppo')
parser.add_argument('--rl_value_loss_coeff', type=float, default=0.2, help='coefficient for critic loss')
parser.add_argument('--rl_seed', type=int, default=-999, help='for fair comparison')
parser.add_argument('--rl_patience', type=int, default=10, help='early stopping')
parser.add_argument('--rl_mini_batch', type=int, default=500, help='in rl setting')
parser.add_argument('--rl_plan_depth', type=int, default=1, help='plannign in AC D-PG')

"""
train2 combinations:
[dpg || pg || ppo] + [crit, wn] 
[dpg || ac_dpg] + [stable_q]
[ppo_fl] + [crit]
[dpg] + [stable_q_fix]

"""

args = parser.parse_args()

if 'M' or 'v' in args.feat:
    args.max_len = 100
else:
    args.max_len = args.n*5

torch.set_printoptions(precision=15)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.rl_seed != -999:
    np.random.seed(args.rl_seed)
    torch.manual_seed(args.rl_seed)
    torch.cuda.manual_seed(args.rl_seed)
    random.seed(args.rl_seed)
    torch.cuda.manual_seed_all(args.rl_seed)


# W&B configuration
if args.wandb:
    import wandb
    wandb.init(project=args.train2, name=str(args.job))
    wandb.config.update(args)

args.nmotifs = 1

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

timestamp = datetime.now().strftime("%mm%dd_%H%M_") + str(args.job)
print(timestamp)

if args.wandb:
        wandb.log({'tstamp':timestamp})

# motif: ratio = 1: 1:50, 2: 1:100, 3: 1:500, 4: 1:1000
# choose 2,4,5,6,7
if args.mtype == 'm':
    if args.motif == 1:
        all_motifs = {10:'1111111', 30:'1000101111', 50:'10001010001'}
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
elif args.mtype == 'mult':
    if args.motif == 3:
        all_motifs = {30:'multipl_3'}
    elif args.motif == 17:
        all_motifs = {30:'multipl_17'}
        
#wandb.config.update({'motif':all_motifs[args.n]})

entp_motifs_tm = {10:{'m.1111111':2.995732273553991/11}, 30:{'mult.multipl_3':19.62365305094772/31, 'mult.multipl_17':17.889051994558614/31, 'mam.100010100011':16.282530254126048/31, 'mam.1000101111100011':13.57540128031525/31,
                    'm.10001010001':16.15303451776991/31,'m.10001011111000':13.923144487457433/31,
                 'mam.10001011111000':14.935250784153713/31, 'm.01011101101':16.1633538708637/31, 
                 'm.001001100111':15.420728378322668/31,'m.1011100111001':14.6736907/31, 'mam.01011101101':16.950563779/31,
                 'mam.001001100111':16.2827152768/31, 'mam.1011100111001':15.61062622/31, 'm.1000101000101':14.66329972621143/31}, 100:{'m.0111010000011101':62.665668876452344/101}}

z_motifs = {10:{'m.1111111':0.01953125}, 30:{'mult.multipl_3':0.3333333343343343, 'mult.multipl_17':0.05882352952952953, 'mam.100010100011':0.0046360, 'mam.1000101111100011':0.00022888,
                    'm.10001010001':0.00964437,'m.10001011111000':0.0010371580,
                 'mam.10001011111000':0.001037158, 'm.01011101101':0.0097444, 
                 'm.001001100111':0.004637, 'm.1011100111001':0.002196863, 'mam.01011101101':0.00974440,
                 'mam.001001100111':0.004637, 'mam.1011100111001':0.002196863, 'm.1000101000101':0.0021741539239883423}, 100:{'m.0111010000011101':0.0012952530732785747}}

entp_motifs = {}

for ni, m in all_motifs.items():
    if ni in entp_motifs_tm and ni == args.n:
        entp_motifs[ni] = entp_motifs_tm[ni][args.mtype+'.'+m.split('.')[0]]


# get data 

def get_batch(source_data, batch):
    data = source_data[:-1,batch:batch+batch_size]
    target = source_data[1:,batch:batch+batch_size].contiguous().view(-1)
    return data, target

def get_batch_fsz(source_data, batch):
    data = source_data[:-1,batch:batch+batch_size]
    target = source_data[1:,batch:batch+batch_size].contiguous()
    return data, target

def load_data_mult(n, sz, motif, ds_type):
    ds = []
    # input: <bos> binary string <eos>
    # 3 {0,1}^n 2
    data_file =  os.path.join(os.path.join(args.logdir,'data'), 'multipl_%s'%(args.motif),"%s.txt"%ds_type)
    max_len = 0
    
    with open(data_file, "r") as file:
        for line in file:
            #assert motif in line
            ds += [line.strip()]
            max_len = max(max_len, len(line.strip()))
            #print(line.strip())
            if len(ds)>=sz:
                break
    n = max_len
    args.n = max_len

    original = ''
    for l in ds:
        original += ' '+ ''.join(c+' ' for c in l).strip()
        original += ' 2 '+ ''.join(str(PAD)+' ' for _ in range(max_len-len(l))).strip()
        original = original.strip()
   
    print(len(original), max_len)
    n += 1
    original = np.fromstring(original, dtype=int, sep=' ')
    original = original.reshape((original.shape[0]//n, n)).transpose()
    
    for i in range(original.shape[1]):
        res = ''.join([str(original[j,i]) for j in range(original.shape[0])])
        #assert flag
    
    dataset = (np.ones((n+1, original.shape[1]))).astype(int)
    dataset[1:] = original
    dataset[0] = dataset[0]*3
    #dataset[-1] = dataset[-1]*2
    print(dataset.shape, batch_size)
    assert dataset.shape[1] >= sz
    ds = dataset[:, :batch_size*int(1.0*dataset.shape[1]/batch_size)]
    return torch.from_numpy(ds).cuda()


def load_data_motif(n, sz, motif, ds_type):
    ds = ""
    # input: <bos> binary string <eos>
    # 3 {0,1}^n 2
    if args.nmotifs == 1:
        data_file = os.path.join(os.path.join(args.logdir,'data'), 'pfsa_%d_%s'%(n, motif),"%s.txt"%ds_type)
    else:
        data_file = os.path.join(os.path.join(args.logdir,'data'), 'pfsa_%d_%s'%(n-1, motif),"%s.txt"%ds_type)
    
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


# ------------------------------------------------------------------------
# ------------ classes: RNN, GAMs, WhiteNoise with filter ----------------

def repackage_hidden(h):
    """detach vars from their history."""
    return tuple(Variable(h[i].data) for i in range(len(h)))  

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# some part of the language model architecture from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModel/
class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, policy=False, policy_log=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        
        self.encoder = nn.Embedding(ntoken, ninp)
        # 0 1 <EOS> <BOS> PAD
        one_hot_vecs = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0], [0,0,0]])
        self.encoder.weight.data.copy_(torch.from_numpy(one_hot_vecs))
        self.freeze_layer(self.encoder)
        
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout) 
        # <bos> is not in the output vocabulary       
        self.decoder = nn.Linear(nhid, ninp)
        self.policy = policy

        if policy and ('crit' in args.train2 or 'ac_dpg' in args.train2):
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
            # spit out values of Z(s) leaves
            if 'ac_dpg_a' in args.train2:
                zn_out = ninp
            else:
                zn_out = 1

            if policy_log:
                # log_Z(s)
                self.critic = nn.Sequential(init_(nn.Linear(nhid, nhid)), nn.Tanh(), init_(nn.Linear(nhid, zn_out)))
            else:
                # Z(s)
                self.critic = nn.Sequential(init_(nn.Linear(nhid, nhid)), nn.Tanh(),  init_(nn.Linear(nhid, zn_out)), nn.ReLU())

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

    def forward(self, input, hidden, len_inp, mask, critic=False):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output) # [seq_len ,batch, nhid]
        # [seq_len*batch, ntok]
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = torch.mul(decoded.view(output.size(0), output.size(1), decoded.size(1)), mask)

        if self.policy and ('crit' in args.train2 or 'ac_dpg' in args.train2) and critic:
            est_z = self.critic(output.view(output.size(0)*output.size(1), output.size(2)))
            est_z = torch.mul(est_z.view(output.size(0), output.size(1), est_z.size(1)), mask)
            return decoded, hidden, est_z
        else:
            return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

class White_noise_filter(nn.Module):
    # biased white noise with filter for strings
    # of length!=n and not containing the motif
    def __init__(self, probs,  feat, motifs):
        super(White_noise_filter, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.feat = feat
        self.motifs = motifs
        
        self.encoder = nn.Embedding(ntoken, 1)
       
        one_hot_vecs = np.array([[pi] for pi in probs+[1, 1]])
        self.encoder.weight.data.copy_(torch.from_numpy(one_hot_vecs))
        # probs for: 0 1 <EOS> <BOS> PAD
        self.probs = torch.tensor(probs).cuda()
        self.freeze_layer(self.encoder)
    
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def init_hidden(self, bsz):
        return (None, None)
    
    def forward(self, input, hidden, len_tar, mask):
        # [seq x batch x 1]
        probs = self.encoder(input)

        # 1 = no motif
        x_feat = get_features(input, self.motifs, self.feat)[:,0]
        if  'no_motif' in args.debug_opt:
            x_feat = x_feat*0
            
        log_lin = 0
        #len_tar, _,_,_ = get_length_mask(input)
        # 1 = length different from n
        if 'i' in args.feat:
            len_feat = (torch.abs(len_tar-(args.n+1))>=10).float()
        else:
            x_feat += (len_tar!=(args.n+1)).float()
        
        infs =  -torch.ones(probs.size(1)).cuda()*float('Inf')
        if 'rew1' in args.debug_opt:
            logits =  torch.zeros(probs.size(1)).cuda()
        else:
            logits = torch.log(probs).sum(0).squeeze()

        if 'i' in args.feat:
            log_05 = np.log(0.5)
            logits = torch.where(((x_feat==0) | (x_feat==0.5)) & (len_feat==0), logits, infs)
            logits = torch.where(((x_feat==0.5) & (len_feat==0))|((x_feat==0) & (len_tar!=(args.n+1))), logits+log_05, logits)
            if np.random.rand()<0.001:
                print('X', input[:,:5], 'feat', x_feat[:5], logits[:5])
        else:
            x_feat = torch.clamp(x_feat, min=0, max=1)
            # if all features are on - use logits, else prob is 0
            logits = torch.where(x_feat==0, logits, infs)
        #print(logits[:10].data.cpu().numpy(), len_tar[:10],  (len_tar!=(args.n+1))[:10], args.n+1)
        # [batch]
        return logits.unsqueeze(1), None, log_lin

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
            # correlated features
            if j < len(args.feat)-4:
                if args.nmotifs == 1:
                    if j == len(args.feat)-7:
                        # motif
                        if args.mtype == 'm':
                            out += [1 - int(motifs[i] in s)]
                        elif args.mtype == 'mult':
                            if s[0] == '3':
                                digits = s[1:]
                            else:
                                digits = s
                            end_idx = digits.find('2')
                            if end_idx != -1:
                                digits = digits[:end_idx]
                            if digits:
                                out += [1-int(int('0b'+digits,2)%args.motif == 0 and int('0b'+digits,2)!=0)]
                            else:
                                out += [1]
                    elif j == len(args.feat)-6:
                        # supermotif
                        motif_j = motifs[i] + '0'*1
                        out += [1 - int(motif_j in s)]
                    elif j == len(args.feat)-5:
                        # submotif
                        motif_j = motifs[i][:len(motifs[i])//2]
                        out += [1 - int(motif_j in s)]
                elif args.nmotifs == 2:
                    if j in [j == len(args.feat)-8, j == len(args.feat)-6]:
                        # motif
                        out += [1 - int(motifs[max(0, j-1)] in s)]
                    elif j in [j == len(args.feat)-7, j == len(args.feat)-5]:
                        # submotif
                        motif_j = motifs[max(0, j-2)][:len(motifs[max(0, j-2)])//2]
                        out += [1 - int(motif_j in s)]
            else:
                # distractive features
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
            out += [get_edit_frc(s, motifs[i])]
        elif feat[i][j] == 's':
            out += [get_longestsubstr_frc(s, motifs[i])]
        elif feat[i][j] == 'l':
            out += [int(np.abs(len(s)-args.n-1)>=3)]

        elif feat[i][j] == 'M':
            out += [(len(s)*1.0)/args.max_len]
        elif feat[i][j] == 'v':
            out += [(1.0*len(s)**2)/(args.max_len**2)]

        elif feat[i][j] == 'm':
            out += [get_longestsubstr_frc(s, motifs[i]) + get_edit_frc(s, motifs[i])]
        elif feat[i][j] == 'i':
            val = get_longestsubstr(s, motifs[i])
            if np.abs(val-len(motifs[i]))==0:
                out += [0]
            elif np.abs(val-len(motifs[i]))<=3:
                out += [0.5]
            else:
                out += [1]
            if np.random.rand()<0.00001:
                print('X', s, 'val', val, np.abs(val-len(motifs[i])), out)
    return out

def get_longestsubstr(s, motif):
    
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
    return max_lss

def get_longestsubstr_frc(s, motif):

    max_lss = get_longestsubstr(s, motif)
    return 1-(1.*max_lss)/len(motif)

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

def argmax_quadratic(left,right,a,b):
    '''
    Find the argmax of $ax^2 + bx$ on the interval [left,right]
    '''
    if a < 0:
        global_argmax = -b/(2*a)
        if left < global_argmax and global_argmax < right:
            return global_argmax
        else:
            return np.argmax([a*left**2 + b*left, a*right**2 + b*right])
    else:
        return np.argmax([a*left**2 + b*left, a*right**2 + b*right])
    



# -----------------------------------------------
# -------------------- utils --------------------

def to_one_hot(y, n_dims=None):
    """ Take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else ninp
    y_one_hot = torch.zeros(y_tensor.size(0), n_dims).scatter_(1, y_tensor, 1).cuda()
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def get_log_r(r_output, ce_target, mask_tar, ce_criterion):
    # get logits from the AMs output layer for a specific target sequence
    # r_output: [seq_len x batch x ninp] -> log_r_seq: [seq_len x batch x 1]
    # ce_target: [seq_len x batch]; indices

    # mask PAD symbol to keep short output vocabulary
    ce_target = torch.mul(ce_target.float(), mask_tar[:,:,0]).long()
    # [(n+1) x batch x 1]
    r_output = torch.nn.functional.log_softmax(r_output, dim=2)
    log_r_seq = torch.sum(r_output.view(-1, ninp) * to_one_hot(ce_target.view(-1)), dim = 1)
    log_r_seq = torch.mul(log_r_seq.view(mask_tar.size()), mask_tar)
    # [seq_len x batch x 1]
    return log_r_seq

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def init_rnn_from_proposal(model_r, policy_log, policy):
    # copy model_r to model_q
    model_q = RNNModel(ntoken, ninp, nhid, nlayers, dropout, policy=policy, policy_log=policy_log)
    model_q.cuda()
    pretrained_dict = model_r.state_dict()
    model_dict = model_q.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model_q.load_state_dict(model_dict)

    return model_q

def cat_variable_length(a, b):
    seq_len = max(a.size()[0], b.size()[0])
    if a.size()[0] < seq_len:
        padding = torch.ones((seq_len-a.size()[0], a.size(1))).long()*PAD
        if a.is_cuda:
            padding = padding.cuda()
        a = torch.cat((a, padding), dim=0)
    if b.size()[0] < seq_len:
        padding = torch.ones((seq_len-b.size()[0], b.size(1))).long()*PAD
        if a.is_cuda:
            padding = padding.cuda()
        b = torch.cat((b, padding), dim=0)

    return torch.cat((a,b), dim=1)

def isfinite(x):
    """
    Quick pytorch test that there are no nan's or infs.
    
    note: torch now has torch.isnan
    url: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519
    """
    not_inf = ((x + 1) != x)
    not_nan = (x == x)
    return not_inf & not_nan

def sample_from_rnn(model):
    n = args.n
    motifs = all_motifs[args.n].split('.')
    batch_size = 5000
    x, log_pi, inp, len_inp, action, mask_tar = sample_data_inp_targ_vary(model, 
                                                    batch_size, max_len=500)
    avg_len = np.round(len_inp.float().mean().data.cpu().numpy(),decimals=1)
    print('avg len ', avg_len)
    x = x.data.cpu().numpy()
    count = 0     
    for i in range(x.shape[1]):
            res = ''.join([str(x[j,i]) for j in range(x.shape[0])])
            curr_count = 0
            if args.mtype == 'mult':
                if res[0] == '3':
                    digits=res[1:]
                else:
                    digits=res
                end_idx = digits.find('2')
                if end_idx != -1:
                    digits = digits[:end_idx]
                if digits:
                    curr_count += int(int('0b'+digits,2)%args.motif == 0 and int('0b'+digits,2)!=0)
            else:
                for motif in motifs:
                    if motif in res:
                        curr_count += 1
            count += min(1, curr_count)

    print('%d motifs in total %d' % (count, x.shape[1]))
    motif_freq = (1.0*count)/x.shape[1]
    return motif_freq, avg_len

def logsumexp(x, dim=None):
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + numpy.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim)) 






# -----------------------------------------------
# -------------- sampling from LM ---------------


def sample_lm_vary(model, batch_size_i, max_len=None, critic=False): 
    # sample strings of varying length
    # output: [ seq_len x batch ]
    model.eval() 
    # [ 1 x batch ]
    # <pad> idx = 4
    if not max_len:
        max_len=args.n*2+1

    EOS = 2; BOS = 3
    out = [(torch.ones(1)*BOS).cuda().long()]*batch_size_i # contains sequences of variable lenght
    symb = start_symbol[:,:batch_size_i]
    hidden = model.init_hidden(batch_size_i)
    len_inp = torch.ones((batch_size_i), dtype=torch.int64).cuda()
    mask = torch.ones((1, batch_size_i, 1)).cuda()
    all_logits = torch.ones((0, batch_size_i, ninp)).cuda()
    if critic:
        all_z = torch.zeros((0, batch_size_i, 1)).cuda()
    
    for i in range(max_len):
        # [1 x batch x ntok]
        if critic:
            logits, hidden, est_z = model(symb, hidden, len_inp, mask, critic=critic)
        else:
            logits, hidden = model(symb, hidden, len_inp, mask)[:2]
        probs = softm(logits)
        cat_dist = torch.distributions.Categorical(probs=probs)
        # [ 1 x batch ]
        symb = cat_dist.sample()
        flag = False
        
        for b in range(batch_size_i):
            # if the sequence has not terminated yet
            if i==0 or (i>0 and out[b][-1] != EOS): 
                out[b] = torch.cat((out[b], symb[:1,b]), dim=0)
                flag = True
        if not flag:
            break
        # TODO: instead of cat write into predefined array
        all_logits = torch.cat((all_logits, logits), dim=0)
        if critic:
            all_z = torch.cat((all_z, est_z), dim=0)
        
    out = torch.nn.utils.rnn.pad_sequence(out, batch_first=False, padding_value=PAD) 
    model.train()  
    # <bos> 010010101 <eos>
    if critic:
        return out, all_logits, all_z
    else:
        return out, all_logits

def sample_lm_vary_new(model, batch_size_i, max_len=None, critic=False): 
    # sample strings of varying length
    # out: [ seq_len+1 x batch ]
    # all_logits: [ seq_len x batch x ninp ]
    # all_z: [ seq_len x batch x 1]


    model.eval() 
    if not max_len:
        max_len=args.n*2+1
    EOS = 2; BOS = 3
    
    symb = start_symbol[:,:batch_size_i]
    hidden = model.init_hidden(batch_size_i)
    len_inp = torch.ones((batch_size_i), dtype=torch.int64).cuda()
    mask = torch.ones((1, batch_size_i, 1)).cuda()

    all_logits = torch.ones((max_len, batch_size_i, ninp)).cuda()
    out = torch.ones((max_len+1, batch_size_i)).cuda().long()
    out[0,:] = out[0,:]*BOS

    if critic:
        all_z = torch.zeros((max_len, batch_size_i, 1)).cuda()
    
    for i in range(max_len):
        # [1 x batch x ntok]
        if critic:
            logits, hidden, est_z = model(symb, hidden, len_inp, mask, critic=critic)
        else:
            logits, hidden = model(symb, hidden, len_inp, mask)[:2]
        probs = softm(logits)
        cat_dist = torch.distributions.Categorical(probs=probs)
        # [ 1 x batch ]
        symb = cat_dist.sample()
        
        out[i+1:i+2] = symb[:1]
        
        all_logits[i:i+1] = logits
        if critic:
            all_z[i:i+1] = est_z
    
    max_seq_len = 0

    for b in range(batch_size_i):
        ends = (out[:,b] == EOS).nonzero()
        if ends.size(0) == 0: 
            continue # string does not contain EOS

        idx = ends[0,0]
        if idx == max_len:
            max_seq_len = max_len
        
        out[idx+1:,b] = out[idx+1:,b]*0 + PAD
        all_logits[idx:,b] = all_logits[idx:,b]*0 
        if critic:
            all_z[idx:,b] = all_z[idx:,b]*0 

        max_seq_len = max(max_seq_len, idx)
     
    out = out[:max_seq_len+1]
    all_logits = all_logits[:max_seq_len]
    if critic:
        all_z = all_z[:max_seq_len]

    model.train()  
    # <bos> 010010101 <eos>
    if critic:
        return out, all_logits, all_z
    else:
        return out, all_logits


def sample_lm_vary_hid(model, batch_size_i, max_len=None, critic=False):
    # output: [ seq_len x batch ]
    model.eval()
    # [ 1 x batch ]
    # <pad> idx = 4
    if not max_len:
        max_len=args.n*2+1
    out = [(torch.ones(1)*3).cuda().long()]*batch_size_i # contains sequences of variable lenght
    symb = start_symbol[:,:batch_size_i]
    hidden = model.init_hidden(batch_size_i)
    len_inp = torch.ones((batch_size_i), dtype=torch.int64)
    mask = torch.ones((1, batch_size_i, 1)).cuda()
    all_logits = torch.ones((0, batch_size_i, ninp)).cuda()
    hids =  torch.zeros(model.nlayers, 0, model.nhid)
    c_hids = torch.zeros(model.nlayers, 0, model.nhid)

    if critic:
        all_z = torch.zeros((0, batch_size_i, 1)).cuda()
    
    for i in range(max_len):
        # [1 x batch x ntok]
        if critic:
            logits, hidden, est_z = model(symb, hidden, len_inp, mask, critic=critic)
        else:
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
        if not flag:
            break
        hids =  torch.cat((hids, hidden[0].cpu()), dim=1).detach()
        c_hids =  torch.cat((c_hids, hidden[1].cpu()), dim=1).detach()
        all_logits = torch.cat((all_logits, logits), dim=0)
        if critic:
            all_z = torch.cat((all_z, est_z), dim=0)
    
    out = torch.nn.utils.rnn.pad_sequence(out, batch_first=False, padding_value=PAD) 
    # <bos> 010010101 <eos>
    model.train()
    if critic:
        return out, all_logits, hids, c_hids, all_z
    else:
        return out, all_logits, hids, c_hids


def sample_wn(model, batch_size_i, max_len=None):
    # sample strings of varying length from white noise model
    # output: [ seq_len x batch ]
    
    # [ 1 x batch ]
    # <pad> idx = 4
    if not max_len:
        max_len=args.n*2+1
    out = [(torch.ones(1)*3).cuda().long()]*batch_size_i # contains sequences of variable lenght
    all_logits = torch.ones((0, batch_size_i, ninp)).cuda()

    # [1 x batch x ntok]
    probs = model.probs.repeat(1, batch_size_i).view(1, -1, ninp)
    logits = torch.log(probs)

    for i in range(max_len):

        cat_dist = torch.distributions.Categorical(probs=probs)
        # [ 1 x batch ]
        symb = cat_dist.sample()
        flag = False
        for b in range(batch_size_i):
            # if the sequence has not terminated yet
            if i==0 or (i>0 and out[b][-1] != 2): 
                out[b] = torch.cat((out[b], symb[:1,b]), dim=0)
                flag = True
        if not flag:
            break
        all_logits = torch.cat((all_logits, logits), dim=0)
        
    out = torch.nn.utils.rnn.pad_sequence(out, batch_first=False, padding_value=PAD) 

    x = out; log_pi = all_logits
    len_inp, mask_tar, inp, targets = get_length_mask(out)  
    # <bos> 010010101 <eos>
    return x, log_pi, inp, len_inp, targets, mask_tar


softm = nn.Softmax(dim=2)


def sample_data_inp_targ_vary_hid(model, batch_size_i, max_len=None, critic=False):
    # padded variable lengths sequences
    # step by step
    # [ 1 x seq_len*batch ]
    if not max_len:
        max_len = args.n*2+1

    # x:       [seq_len x batch]
    # log_pi:  [seq_len x batch x ninp]
    # hids:    [nlayers x batch*seq_len x nhid]
    # est_z:   [(n+1) x batch x 1]
    if critic:
        x, log_pi, hids, c_hids, est_z  = sample_lm_vary_hid(model, batch_size_i, max_len, critic=critic)
    else:
        x, log_pi, hids, c_hids  = sample_lm_vary_hid(model, batch_size_i, max_len)
    len_inp = (x!= PAD).sum(0)
    len_inp, perm_idx = len_inp.sort(0, descending=True)
    len_inp = len_inp - 1
    x = x[:, perm_idx]
    inp = x[:-1,:]
    log_pi = log_pi[:,perm_idx]
    hids = hids.view(model.nlayers, inp.size(0), inp.size(1), model.nhid)[:, :, perm_idx]
    hids = hids.view(model.nlayers, inp.size(0)*inp.size(1), model.nhid)
    c_hids = c_hids.view(model.nlayers, inp.size(0), inp.size(1), model.nhid)[:, :, perm_idx]
    c_hids = c_hids.view(model.nlayers, inp.size(0)*inp.size(1), model.nhid)
    if critic:
        est_z = est_z[:,perm_idx]

    # [(n+1) x batch]
    targets = x[1:,:] 
    mask_tar = (targets != PAD).unsqueeze(2).float().cuda()   
    len_tar = (targets != PAD).sum(0)
    if critic:
        return x, log_pi, inp, len_inp, targets, mask_tar, hids, c_hids, est_z
    else:
        return x, log_pi, inp, len_inp, targets, mask_tar, hids, c_hids

def sample_data_inp_targ_vary(model, batch_size_i, max_len=None, critic=False):  ###TODO
    # padded variable lengths sequences
    # [ seq_len x batch ]
    if not max_len:
        max_len = args.n*2+1
    if critic:
        # est_z: [(n+1) x batch x 1]
        x, log_pi, est_z = sample_lm_vary(model, batch_size_i, max_len, critic=critic)
    else:
        x, log_pi = sample_lm_vary(model, batch_size_i, max_len)
    len_inp = (x!= PAD).sum(0)
    len_inp, perm_idx = len_inp.sort(0, descending=True)
    len_inp = len_inp - 1
    x = x[:, perm_idx]
    inp = x[:-1,:]
    log_pi = log_pi[:,perm_idx]
    if critic:
        est_z = est_z[:,perm_idx]

    # [(n+1) x batch]
    targets = x[1:,:] 
    mask_tar = (targets != PAD).unsqueeze(2).float().cuda()   
    len_tar = (targets != PAD).sum(0)
    if critic:
        return x, log_pi, inp, len_inp, targets, mask_tar, est_z
    else:
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






# -----------------------------------------------
# -------------- RS and SNIS --------------------

def upper_bound(params):
    # for rejection sampling
    out = log_upper_bound(params)
    return torch.exp(out)

def log_upper_bound(params):
    # for rejection sampling
    # Q(x) = beta*r(x) >= exp(lambda*phi(x))*r(x) = P_lambda(x), all x
    # beta >= exp(lambda*phi(x)), all x
    # linear for all features except x and x**2 feats
    out = torch.zeros((1)).cuda()
    i = 0
    while i < params.size(1):
        # for the length feature - combine two coefficients
        if 'M' == args.feat[i] and 'v' == args.feat[i+1]:
            a = params[0,i+1]
            b = params[0,i]
            x = argmax_quadratic(0,1, a, b)
            y = a*x**2 + b*x
            out = torch.cat((out, y.unsqueeze(0)), dim=0)
            i += 2
        else:
            assert args.feat[i] not in ['M', 'v']
            curr = params[:1,i]
            if curr > 0:
                out = torch.cat((out, curr), dim=0)
            i += 1
    return out.sum(0)

def rejection_sampling(model, ce_criterion, motifs, feat, ro_stats):
    # q(x)=r(x), Q(x)=q(x)*beta>=P_lambda(x)=r(x)*loglin for any x 
    # sample from LM: x ~ q(x)
    # accept with probability ro = P_lambda(x)/Q(x)

    nfeat = sum([sum([int(e!='0') for e in el]) for el in feat])
    samples = torch.ones((1, nfeat)).cuda()
    batch_size_i = 2*batch_size
    
    acceptance_rate, total_samples, accepted = ro_stats
    
    while samples.size(0) <= nsamples:            
        x, log_pi, inp, len_inp, targets, mask_tar = sample_data_inp_targ_vary(model, batch_size_i)

        hidden = model.init_hidden(batch_size_i)
        # log_lin  [ batch x 1 ]        
        r_output, _, log_lin = model(inp, hidden, len_inp, mask_tar) # outpt [seq_len ,batch, ntok]
        # [ batch x 1 ]
        #log_r = get_log_r(r_output, targets, log_lin, mask_tar, ce_criterion)
        #P_lambda = torch.exp(log_r + log_lin)

        # upper bound: P_lambda <= q(x)*exp(max(lambda * feat))
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

    # upper bound: P_lambda <= q(x)*exp(max(lambda * feat))
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




# ------------------------------------------------
# -------------- get cross-entropy ---------------

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
        loss = torch.div(torch.mul(loss, mask_tar).sum(0).squeeze(), len_tar.float()).mean()
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
        len_tar, mask_tar, data, targets = get_length_mask(source_data[:,batch:batch+batch_size])
        batch_size_i = data.size()[1]
        hidden = model.init_hidden(batch_size_i)
        
        r_output, hidden, log_lin = model(data, hidden, len_tar, mask_tar)
        output_flat = r_output.view(-1, ninp)
        targets = torch.mul(targets.float(), mask_tar[:,:,0]).long()
        curr_loss =  torch.mul(ce_criterion(output_flat, targets.view(-1)).view(mask_tar.size()), mask_tar).squeeze().sum(0)
        curr_loss = torch.div(curr_loss, len_tar.float()).mean()
        total_loss += curr_loss.data.float()

    return total_loss / len(batches_id)

def estimate_partition_mc(model, ce_criterion):
    # using importance sampling
    # Z_lambda = E_{x~q(.)}[P_lambda(x)/q(x)] = E_{x~q(.)}[exp(lambda^T feat(x))]
    # sample from q(x) = r(x), use IS to compute expectation w.r.t. p_lambda distribution
    # compute expectation using MC samples 

    batch_size_i = 6500
    N = 160
    z_samples = 0
    for _ in range(N):   
    
        if 'wn' in args.train2:
            x, log_pi, inp, len_inp, target, mask_tar = sample_wn(model, batch_size_i)
        else:
            x, log_pi, inp, len_inp, target, mask_tar = sample_data_inp_targ_vary(model, batch_size_i)
        
        hidden = model.init_hidden(batch_size_i)
        r_output, _, log_lin = model(inp, hidden, len_inp, mask_tar) # outpt [seq_len ,batch, ntok]
        # [ batch x 1 ]
        if 'wn' in args.train2:
            # for filtered white noise
            z_samples += (torch.exp(r_output)!=0).float().mean()
        else:
            z_samples += torch.exp(log_lin).mean()
    
    return z_samples/N 
         



# ------------------------------------------------
# -------------- train LM using CE ---------------

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

def train_r(model, criterion, epoch, source_data, lr, optimizer):
    # train proposal r using CE w.r.t. D
    
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



############################# TRAINING-1 #############################
# get proposal r using D
# obtain P_lambda

def training_1():

    Epoch_start_time = time.time() 

    global batch_size
    n = args.n
    # ----------------------------------------------- train r on original dataset D ------------------------------------------
    batch_size = min(args.ds_size, 500)
    lr = 0.001
    log_dir = os.path.join(args.logdir,'pg_methods/runs/chpt_%s'%(timestamp))
    os.mkdir(log_dir)
    train_size = size = args.ds_size  
    feat = [args.feat]
    motifs = all_motifs[args.n].split('.')
    print('motif %s'%motifs[0])

    # (seq_len, nbatch)
    test_size = 5000
    valid_size = min(max(batch_size, int(0.25*train_size)), 2000)

    if args.mtype == 'mult':
        train_data_D = train_data = load_data_mult(n, train_size, all_motifs[args.n], 'train')
        valid_data_V = valid_data = load_data_mult(n, valid_size, all_motifs[args.n], 'valid')
        test_data  = load_data_mult(n, test_size, all_motifs[args.n], 'test')
    else:
        train_data_D = train_data = load_data_motif(n, train_size, all_motifs[args.n], 'train')
        valid_data_V = valid_data = load_data_motif(n, valid_size, all_motifs[args.n], 'valid')
        test_data  = load_data_motif(n, test_size, all_motifs[args.n], 'test')
    train_feat = get_features(train_data, motifs, feat).mean(0)
    
    print('orig train ds feat = ', train_feat.cpu().numpy())
    print('orig valid ds feat = ', get_features(valid_data,motifs, feat).mean(0).cpu().numpy())
    print('orig test ds feat = ', get_features(test_data,motifs, feat).mean(0).cpu().numpy())
    if args.wandb:
        wandb.log({'true_data_feats':  train_feat.cpu().numpy(), 'test_data_feats':  get_features(test_data,motifs, feat).mean(0).cpu().numpy()})

    best_val_loss = None
    counter = 0
    patience = 8

    writer = SummaryWriter(log_dir=log_dir)
    model_r = RNNModel(ntoken, ninp, nhid, nlayers, dropout)
    model_r.cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    optimizer_r = optim.Adam(model_r.parameters(), lr=lr)

    for epoch in range(1, 100):
        epoch_start_time = time.time()
        train_r(model_r, criterion, epoch, train_data, lr, optimizer_r)
        val_loss = evaluate(model_r, criterion, valid_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} |'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss))
        if args.wandb:
            wandb.log({'epoch': epoch, 'r_valid_ce': val_loss})
        print('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(log_dir,'chpt_%s_r.pt'%(timestamp)), 'wb') as f:
                torch.save(model_r, f)
            best_val_loss = val_loss
            counter = 0
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if best_val_loss:
                counter += 1
                if counter >= patience:
                    break
    del model_r 

    best_val_loss_r = best_val_loss
    model_r = torch.load(os.path.join(log_dir,'chpt_%s_r.pt'%(timestamp)))
    test_loss = evaluate(model_r, criterion, test_data)
    if args.wandb:
        wandb.log({'r_test_ce': test_loss})
    if args.tensorboard:
        writer.add_scalar('data/r_test_ce', test_loss, 0)

    train_loss = evaluate(model_r, criterion, train_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {} | '
          'test ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                     test_loss, math.exp(min(test_loss, 20))))
    print('-' * 89)
    
    entp = entp_motifs[n]
    print('\nTheoretical entp = {} for n = {:2d}  \n'.format(entp,  n))
    if args.wandb:
            wandb.log({'theor_ent': entp})

    test_ce_r = test_loss.float().cpu().numpy().item(0)

    epochs = args.epochs

    # ------------------------------------- get P_lambda -----------------------------------------

    if not 'cyclic' in args.train2 and not 'wn' in args.train2:
        

        lr0 = lr = 10.#0.001
        train_size = size = args.ds_size  
        print('motifs ', motifs)
        
        
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

        am_samples = torch.zeros((args.n+2,0)).long().cuda()              
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            if args.theta_fixed:
                ro_stats, mean_feat, am_samples = cyclic_train_lambda(model_plambda, ce_criterion, epoch, 
                                                train_data_D, lr, motifs, feat, ro_stats, optimizer_pl, train_feat, writer, hash_train, am_samples)

            #val_loss = evaluate_ce_pl_ds(model_plambda, ce_criterion, valid_data)/(n+1)
            # early stopping w.r.t. L1 loss between the moments of the P_lambda and D
            l1_feat = torch.abs(train_feat - mean_feat).sum()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | l1_feat {} '.
                  format(epoch, (time.time() - epoch_start_time), l1_feat))
            print('lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy(), model_plambda.lin_lambda.bias.data.squeeze().cpu().numpy())
            if args.wandb:
                wandb.log({'epoch': epoch, 'plambda_l1_feat': l1_feat})
            if args.train == 'rs':
                acceptance_rate, total_samples, accepted = ro_stats
                print('ro', acceptance_rate.data.cpu().numpy(), 'rate', (accepted)/total_samples) 
                #writer.add_scalar('data/accpt_rate', (accepted)/total_samples, epoch)
            print('mean_feat', mean_feat.data.squeeze().cpu().numpy())
            print('-' * 89)
            
            lr = lr0/(1 + epoch)
                            
            if not best_val_loss or l1_feat < best_val_loss:
                with open(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)), 'wb') as f:
                    torch.save(model_plambda, f)
                best_val_loss = l1_feat #val_loss
                counter = 0
            else:
                if best_val_loss:
                    counter += 1
                    if counter >= patience:
                        break
        del model_plambda 
        model_plambda = torch.load(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)))

        print('-' * 89)

        plambda_time = time.time() - start_time
        
        # hybrid model
        z_estim = estimate_partition_mc(model_plambda, ce_criterion)
        test_loss = evaluate_ce_pl_ds(model_plambda, ce_criterion, test_data, z_estim)/(n+1)
        train_loss = evaluate_ce_pl_ds(model_plambda, ce_criterion, train_data, z_estim)/(n+1)
        
        # autoregressive part of the P_lambda
        test_loss_r = evaluate_ce_r(model_plambda, ce_criterion,  test_data)
        train_loss_r = evaluate_ce_r(model_plambda, ce_criterion, train_data)
        if args.wandb:
            wandb.log({'plambda_test_ce': test_loss, 'plambda_z_estim': z_estim, 'lambda': model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy()})
        if args.tensorboard:
            writer.add_scalar('data/plambda_test_ce', test_loss, 0)
            writer.add_scalar('data/plambda_z_estim', z_estim, 0)
            writer.add_histogram('data/lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy(), 0)

        print('\nTheoretical entp = {:5.4f} for n = {:2d}  \n'.format(entp, n))
        print("test_gams {}, train_gams {}, n {}, ds_size {}, motif {}".
            format(test_loss.data.float(), train_loss.data.float(), n, size, motifs))
        print("test_r {}, train_r {}".
            format(test_loss_r.data.float(), train_loss_r.data.float()))
        print('-' * 89)
        print('lambda', model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy())

        #del model
        os.remove(os.path.join(log_dir, 'chpt_%s_pl.pt'%(timestamp)))
        test_ce_pl,lambd =  [test_loss.data.float().cpu().numpy().item(0), 
                    str(list(model_plambda.lin_lambda.weight.data.squeeze(1).cpu().numpy()))]
    else:
        # otherwise get the true P_lambda(x) = wn(x)*F(x)
        model_plambda, lambd, test_ce_pl = [None]*3
        if 'wn' in args.train2:
            if 'bwn' in args.train2:
                probs = [0.59, 0.4, 0.01] 
            else:
                probs = [(1-1./args.n)/2, (1-1./args.n)/2, 1./args.n]
            model_plambda = White_noise_filter(probs, feat, motifs)
            model_plambda.cuda()
        
    theor_ent,tstamp =  [entp, timestamp]
    all_data = [train_feat, train_data, valid_data, test_data]

    return model_r, model_plambda, test_ce_r, test_ce_pl,theor_ent,tstamp,lambd, Epoch_start_time, writer, optimizer_r, all_data




def cyclic_train_lambda(model, ce_criterion, epoch, source_data, lr, motifs, feat, ro_stats, optimizer, target_feat, writer, hash_source, am_samples):
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
            mean_feat, total_w, total_feat = snis(model, ce_criterion, motifs, feat, source_data, hash_source, total_feat, total_w)
            #total_feat = torch.cat((total_feat, curr_feat), dim=0)
            #total_w = torch.cat((total_w, curr_w), dim=0)
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
            
    
    return ro_stats, all_mean_feat/len(batches_id), am_samples

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
        x, log_pi, inp, len_inp, target, mask_tar = sample_data_inp_targ_vary(model, batch_size_i)

        all_feats =  get_features(x, motifs, feat)      
        log_lin = model.lin_lambda(all_feats)

        # upper bound: P_lambda(x) <= q(x)*exp(max(lambda * feat))
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





############################# TRAINING-2 #############################


# -------------------------------------------------------------------
# ------------------------ Distillation ---------------------------------------

def r_plambda_distill_pitheta(model_plambda, model_r, tstamp, Epoch_start_time, writer, all_data):

    n = args.n
    epochs = args.epochs
    log_dir = os.path.join(args.logdir,'pg_methods/runs/chpt_%s'%(timestamp))
    #train_feat, train_data, valid_data, test_data = all_data
    entp = entp_motifs[n]
    patience = 8
    motifs = all_motifs[args.n].split('.')

    train_feat, train_data_D, valid_data_V, test_data = all_data

    train_size = size = args.ds_size  
    feat = [args.feat]
    print('motifs ', motifs)

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

    train_data, ro_stats = cyclic_distill_rejection_sampling(model_plambda, ce_criterion, motifs, feat, ro_stats, size)
    valid_data, ro_stats = cyclic_distill_rejection_sampling(model_plambda, ce_criterion, motifs, feat, ro_stats, valid_size)

    print('train_data', train_data.size(), 'val', valid_data.size())
    
        
    print('-' * 89)
    acceptance_rate, total_samples, accepted = ro_stats
    print('ro', acceptance_rate.data.cpu().numpy(), 'rate', (accepted)/total_samples, 'num', 
                  total_samples, 'accpt', accepted)

    train_feat_pl = get_features(train_data, motifs, feat)
    valid_feat_pl = get_features(valid_data, motifs, feat)
    print('plambda train ds feat = ', train_feat_pl.mean(0).cpu().numpy())
    print('plambda valid ds feat = ', valid_feat_pl.mean(0).cpu().numpy())
    print('orig test ds feat = ', get_features(test_data, motifs, feat).mean(0).cpu().numpy())
    
    mfeat_pl = ((train_feat_pl.sum(0) + valid_feat_pl.sum(0))/(train_feat_pl.size(0)+valid_feat_pl.size(0))).cpu().numpy()
    train_l1_pl = np.absolute(train_feat.data.float().cpu().numpy() - mfeat_pl).item(0)
    

    # ----------------------------------------- train pi_theta on distilled dataset -----------------------------------------
    batch_size = min(args.distill_size, 500)

    lr = 0.001
    best_val_loss = None
    counter = 0

    model_pitheta = RNNModel(ntoken, ninp, nhid, nlayers, dropout)
    model_pitheta.cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    optimizer_pi = optim.Adam(model_pitheta.parameters(), lr=lr)

    # expand D with the distilled dataset
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

        if args.wandb:                
                wandb.log({'epoch': epoch, 'q_val_ce': val_loss})

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)), 'wb') as f:
                torch.save(model_pitheta, f)
            best_val_loss = val_loss
            counter = 0
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if best_val_loss:
                counter += 1
                if counter >= patience:
                    break

    Final_duration = (time.time() - Epoch_start_time)/3600.
    

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

    if args.wandb:
        wandb.log({'pitheta_test_ce': test_ce_pi, 'mfeat_pl_distill':mfeat_pl})

    mfeat_pl = str(list(mfeat_pl))


    writer.close()
    return [test_ce_pi,mfeat_pl,tstamp,Final_duration,train_l1_pl,model_pitheta]

def cyclic_r_plambda_pitheta(model_plambda, model_r, tstamp, Epoch_start_time, writer, optimizer_r, all_data):
    # cyclic mode for distillation

    n = args.n
    train_feat, train_data_D, valid_data_V, test_data_T = all_data
    entp = entp_motifs[n]
    epochs = args.epochs
    log_dir = os.path.join(args.logdir,'pg_methods/runs/chpt_%s'%(timestamp))

    # ------------------------------------- training cycle: P_lambda + pi_theta -----------------------------------------

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
    
    #writer = SummaryWriter(log_dir=log_dir)
    
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

    patience = args.rl_patience
    patience_distl = 5
    patience_dist_pi = 15
    counter_dist= 0
    flag_pi_distill = False

    criterion = nn.CrossEntropyLoss(reduction='none')

    best_distill_acceptance_rate = None


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
            
            test_ce_pl,theor_ent,tstamp,lambd =  [999, entp, timestamp, 
                            str(list(model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy()))]

        # --------------------------------------- distill batch from P_lambda -----------------------------------------------------
        
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
        distill_acceptance_rate = acceptance_rate.data.cpu().numpy().item(0)

        print('-' * 89)
        print('| distill acceptance_rate {} '.format(distill_acceptance_rate))
        print('-' * 89)

        # ---------------   cyclically update pi_theta and P_lambda until the desired acceptance rate is reached --------------------------
        if not flag_pi_distill:
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

            if args.train2 == 'cyclic_1':
                # ---------------------------------------- pi_theta update on one batch -------------------------------------
                
                ce_loss = single_update_r(model_pitheta, train_data, optimizer_r, lr, criterion)
                total_loss_cycl += ce_loss.data.float()


                if b % log_interval == 0 and b > 0:
                    cur_loss = total_loss_cycl / log_interval
                    elapsed = time.time() - start_time
                    print('| iter {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                          'loss {:5.2f} | ppl {:8.5f}'.format(
                              Epoch, b, updates_per_epoch, lr, 
                              elapsed * 1000 / log_interval, cur_loss, math.exp(min(cur_loss, 20))))

            elif args.train2 == 'cyclic_r':
                # ---------------------------------------- retrain pi_theta -------------------------------------
                best_val_loss = None
                counter = 0

                print('retrain pi_theta on the D & distilled dataset')
                current_train_data = cat_variable_length(distilled_D, train_data_D)

                for epoch in range(1, epochs+1):
                    epoch_start_time = time.time()
                    train_r(model_pitheta, criterion, epoch, current_train_data, lr, optimizer_r)
                    val_loss = evaluate(model_pitheta, criterion, distilled_V)
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
                          'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                                     val_loss, math.exp(min(val_loss, 20))))
                    print('-' * 89)

                    if not best_val_loss or val_loss < best_val_loss:
                        with open(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)), 'wb') as f:
                            torch.save(model_pitheta, f)
                        best_val_loss = val_loss
                        counter = 0
                    else:
                        # Anneal the learning rate if no improvement has been seen in the validation dataset.
                        if best_val_loss:
                            counter += 1
                            if counter >= patience_distl:
                                break

                model_pitheta = torch.load(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)))

        if distilled_D.size(1) >= args.distill_size:
            break

        print('distilled_D size', distilled_D.size())

    distilled_D = cat_variable_length(distilled_D, train_data_D)
    print('distilled_D', distilled_D.size())
    print('distilled_V', distilled_V.size())

    #if args.train2 == 'cyclic_1':
    model_pitheta = RNNModel(ntoken, ninp, nhid, nlayers, dropout)
    model_pitheta.cuda()
    optimizer_pi = optim.Adam(model_pitheta.parameters(), lr=lr)


    # ----------------------------------------- train pi_theta on distilled ds -----------------------------------------
    for Epoch in range(Epochs): 
            print('----- epoch %d ------'%Epoch)       

            epoch_start_time = time.time()
            train_r(model_pitheta, criterion, Epoch, distilled_D, lr, optimizer_pi)
            val_loss = evaluate(model_pitheta, criterion, distilled_V)

            if args.wandb:                
                wandb.log({'epoch': Epoch, 'q_val_ce': val_loss})

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
    
    if args.wandb:
        wandb.log({'pitheta_test_ce': test_ce_pi, 'mfeat_pl_distill':mfeat_pl})

    lambd = str(list(model_plambda.lin_lambda.weight.data.squeeze().cpu().numpy()))
    mfeat_pl = str(list(mfeat_pl))

    return [test_ce_pi,mfeat_pl,tstamp,Final_duration,train_l1_pl, model_pitheta, lambd, test_ce_pl]




# ------------------------------------------------------------------------
# ----------------------------- RL ---------------------------------------


def rl_pitheta(model_plambda, model_r, tstamp, Epoch_start_time, writer, all_data):

    n = args.n
    train_feat, _, valid_data, test_data = all_data
    entp = entp_motifs[n]
    epochs = args.epochs
    patience = args.rl_patience
    log_dir =os.path.join(args.logdir,'pg_methods/runs/chpt_%s'%(timestamp))

    lr = args.rl_lr
    best_val_loss = None
    counter = 0
    feat = [args.feat]
    motifs = all_motifs[args.n].split('.')

    criterion = nn.CrossEntropyLoss(reduction='none')

    if args.wandb and not 'wn' in args.train2:
        size = 600
        acceptance_rate = torch.zeros((1)).cuda()
        total_samples = 0
        accepted = 0
            
        ro_stats = [acceptance_rate, total_samples, accepted]
        distilled_data, ro_stats = cyclic_distill_rejection_sampling(model_plambda, criterion, motifs, feat, ro_stats, size)

        print('distilled_data', distilled_data.size())
        
            
        print('-' * 89)
        acceptance_rate, total_samples, accepted = ro_stats
        print('ro', acceptance_rate.data.cpu().numpy(), 'rate', (accepted)/total_samples, 'num', 
                      total_samples, 'accpt', accepted)

        d_feat_pl = get_features(distilled_data, motifs, feat).mean(0).cpu().numpy()
        print('plambda train ds feat = ', d_feat_pl)

        wandb.log({'mfeat_pl_distill':d_feat_pl})

    print('train pi_theta in %s'%args.train2)


    # ----------------------------------------- train pi_theta using policy gradient ----------------------------------------- 

    policy, policy_log = False, False
    if 'crit' in args.train2 or 'ac_dpg' in args.train2:
        policy = True
        #if 'ac_dpg_a' in args.train2:
        policy_log = True
        

    model_pitheta = RNNModel(ntoken, ninp, nhid, nlayers, dropout, policy=policy, policy_log=policy_log)
    model_pitheta.cuda()
    
    
    optimizer_pi = optim.Adam(model_pitheta.parameters(), lr=lr)

    if 'stable_q' in args.train2:
        model_q = init_rnn_from_proposal(model_r, policy_log, policy)
        q_val_ce = evaluate(model_q, criterion, valid_data)
        print('q_val_ce', q_val_ce)
        if args.wandb:
            wandb.log({'epoch': 0, 'q_val_ce': q_val_ce})
        
    else:
        model_q = None


    if 'ac_dpg' in args.train2:
        # use stable new network for critic
        model_crit = init_rnn_from_proposal(model_r, policy_log, policy)
        optimizer_w = optim.Adam(model_crit.parameters(), lr=lr)
        optimizers = [optimizer_pi, optimizer_w]

    if args.wandb:
        val_loss = evaluate(model_pitheta, criterion, valid_data)
        wandb.log({'epoch': 0, 'pitheta_valid_ce': val_loss})

    if 'stable_q' in args.train2 and 'crit' in args.train2:
        print("wrong train2 definition: %s"%args.train2)
        raise

    z_estim_mc = 1 #estimate_partition_mc(model_plambda, criterion)

    print('%s optimizer with lr=%f'%(args.optim, lr))

    batch_size = args.rl_mini_batch
    nbatches = list(range(0, int((1.0*args.distill_size*args.rl_scale_iter)/batch_size)*batch_size, batch_size))

    #if args.wandb:
    #    wandb.watch(model_pitheta)
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
       
        if 'ac_dpg' in args.train2:
            model_pitheta, total_loss, model_q, model_crit = train_pi_theta_ac_dpg(model_pitheta, model_plambda, criterion, epoch, lr, motifs, feat, optimizers, writer, z_estim_mc, model_q, model_crit)
        elif 'ppo_fl' in args.train2:
            model_pitheta, total_loss, model_q = train_pi_theta_ppo_flat(model_pitheta, model_plambda, criterion, epoch, lr, motifs, feat, optimizer_pi, writer, z_estim_mc, model_q)
        elif 'ppo' in args.train2:
            model_pitheta, total_loss, model_q = train_pi_theta_ppo(model_pitheta, model_plambda, criterion, epoch, lr, motifs, feat, optimizer_pi, writer, z_estim_mc, model_q)
        else:
            model_pitheta, total_loss, model_q = train_pi_theta_pg(model_pitheta, model_plambda, criterion, epoch, lr, motifs, feat, optimizer_pi, writer, z_estim_mc, model_q, valid_data)
        val_loss = evaluate(model_pitheta, criterion, valid_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid ce {} '.format(epoch, (time.time() - epoch_start_time),
                                         val_loss))
        print('-' * 89)

        if args.tensorboard:
            writer.add_scalar('pitheta/val_loss_ce', val_loss.mean().data.cpu().numpy(), epoch)
        if args.wandb:
            wandb.log({'epoch': epoch*len(nbatches), 'pitheta_valid_ce': val_loss})

        print('pi_theta validation score: {}'.format(val_loss))

        if 'stable_q' in args.train2:
            
            if args.wandb:                
                wandb.log({'epoch': epoch*len(nbatches), 'q_val_ce': q_val_ce})
            if q_val_ce > val_loss and not 'stable_q_fix' in args.train2:
                model_q = RNNModel(ntoken, ninp, nhid, nlayers, dropout, policy=policy, policy_log=policy_log)
                model_q.cuda()
                model_q.load_state_dict(model_pitheta.state_dict()) 
                q_val_ce = val_loss

        # Save the model if the validation loss is the best we've seen so far.
        if epochs%1 == 0 and args.tensorboard:
            for name, param in model_pitheta.named_parameters():
                writer.add_histogram('pitheta/'+name, param.clone().cpu().data.numpy(), epochs)
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)), 'wb') as f:
                torch.save(model_pitheta, f)
            best_val_loss = val_loss
            counter = 0
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if best_val_loss:
                counter += 1
                if counter >= patience:
                    break

    Final_duration = (time.time() - Epoch_start_time)/3600.
    
    mfeat_pl = 'none'

    del model_pitheta 
    model_pitheta = torch.load(os.path.join(log_dir,'chpt_%s_pi.pt'%(timestamp)))
    test_loss = evaluate(model_pitheta, criterion, test_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {} | '
          'test ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                     test_loss, math.exp(min(test_loss, 20))))
    print('-' * 89)
    if args.tensorboard:
        writer.add_scalar('pitheta/test_loss_ce', test_loss.mean().data.cpu().numpy(), 0) 
    test_ce_pi = test_loss.float().cpu().numpy().item(0)
    writer.close()
    if args.wandb:
        wandb.log({'pitheta_test_ce': test_ce_pi})

    return [test_ce_pi,mfeat_pl,tstamp,Final_duration,999, model_pitheta]



def train_pi_theta_pg(model_pitheta, model_plambda, ce_criterion, epoch, lr, motifs, feat, optimizer_pi, writer, z_estim_mc, model_q, valid_data):
    # D-PG and PG modes
    # in the critic mode: critic shares parameters with policy

    model_pitheta.train()

    total_loss = 0.
    start_time = time.time()
    batch_size = args.rl_mini_batch
    nbatches = list(range(0, int((1.0*args.distill_size*args.rl_scale_iter)/batch_size)*batch_size, batch_size))
    T_loss = 0

    pl_hidden = model_plambda.init_hidden(batch_size)
    acceptance_rate = 0

    # sample x ~ pi_theta(.)
    if 'fix_length' in args.debug_opt:
        max_len = args.n+1
    else:
        max_len = args.max_len #5*args.n


    # --------------------------- bebugging: inspect q ---------------------------------
    if 'stable_q' in args.train2:

        bs = 5000
        x, log_pi, inp, len_inp, action, mask_tar = sample_data_inp_targ_vary(model_q, bs, max_len=max_len)
        pl_hidden_q = model_plambda.init_hidden(bs)
        r_output, _, log_lin = model_plambda(inp, pl_hidden_q, len_inp, mask_tar)
        if not 'wn' in args.train2:
            log_r = get_log_r(r_output, action, mask_tar, ce_criterion).sum(0) # [batch x 1]
        else:
            log_r = r_output
        P_lambda = torch.exp(log_r + log_lin)

        pass_filter = (P_lambda != 0).float().mean()

        log_pi_all = get_log_r(log_pi, action, mask_tar, ce_criterion)
        log_pi_a_q = log_pi_all.sum(0)
        
        if 'dpg' in args.train2:
            #assert P_lambda.size() == log_pi_a.size()
            rewards = torch.exp(log_r + log_lin - log_pi_a_q)/z_estim_mc
        elif 'pg' in args.train2:
            rewards = P_lambda
            if not 'wn' in args.train2:
                rewards = rewards*(2**args.n)
            
        rewards = rewards.detach()

        batches_id = list(range(0, x.size(1)))
        shuffle(batches_id)

        print('inspect q')
        if args.wandb:
            wandb.log({'epoch': epoch*len(nbatches), 'q_avg_len': len_inp.float().mean().data.cpu().numpy(),
                'q_pass_filter': pass_filter.data.cpu().numpy(),'q_a':  wandb.Histogram(torch.exp(log_pi_a_q).data.cpu().numpy()) })
        print('x', x[:,batches_id][:,:10].t().squeeze().data.cpu().numpy())
        print('q_a', torch.exp(log_pi_a_q)[batches_id][:10].squeeze().data.cpu().numpy())
        print('len_inp', len_inp[batches_id][:10].squeeze().data.cpu().numpy())
        print('pass', (P_lambda != 0)[batches_id][:10].squeeze().data.cpu().numpy())
        print('pass_filter', pass_filter.squeeze().data.cpu().numpy())

    # -----------------------------------------------------------------------

    for jj, i in enumerate(nbatches):
        model_pitheta.zero_grad()
        optimizer_pi.zero_grad()
        

        if not 'stable_q' in args.train2:
            model_q = model_pitheta

        if 'crit' in args.train2:
            x, log_pi, inp, len_inp, action, mask_tar, est_z = sample_data_inp_targ_vary(model_q, batch_size, max_len=max_len, critic=True)
            # get values for the last action
            # [seq_len x batch x 1] -> [batch x 1]
            est_z = torch.sum(est_z.squeeze().t().contiguous() * to_one_hot(len_inp-1, n_dims=est_z.size(0)), dim = 1).unsqueeze(1)
            if not 'stable_q' in args.train2:
                hidden = model_q.init_hidden(inp.size(1))
                log_pi, hidden, est_z = model_q(inp, hidden, 0, mask_tar, critic=True)
                len_inp, mask_tar, inp, action = get_length_mask(x)
        else:
            x, log_pi, inp, len_inp, action, mask_tar = sample_data_inp_targ_vary(model_q, batch_size, max_len=max_len)
            if not 'stable_q' in args.train2:
                hidden = model_q.init_hidden(inp.size(1))
                log_pi, hidden = model_q(inp, hidden, 0, mask_tar)
                len_inp, mask_tar, inp, action = get_length_mask(x)

        # get Plambda(x)
        r_output, _, log_lin = model_plambda(inp, pl_hidden, len_inp, mask_tar)
        if not 'wn' in args.train2:
            log_r = get_log_r(r_output, action, mask_tar, ce_criterion).sum(0) # [batch x 1]
        else:
            log_r = r_output
        P_lambda = torch.exp(log_r + log_lin)
        # [(n+1) x batch x 1]
        log_pi_all = get_log_r(log_pi, action, mask_tar, ce_criterion)
        log_pi_a_q = log_pi_all.sum(0)
        
        
        if 'dpg' in args.train2:
            #assert P_lambda.size() == log_pi_a.size()
            rewards = torch.exp(log_r + log_lin - log_pi_a_q)/z_estim_mc
        elif 'pg' in args.train2:
            rewards = P_lambda
            
        rewards = rewards.detach()
        returns = rewards.clone()

        if 'crit' in args.train2:
            assert rewards.size() == est_z.size()
            value_loss = 0.5 * (rewards - est_z).pow(2).mean()
            rewards = rewards - est_z.detach()

        if 'stable_q' in args.train2:
            hidden = model_pitheta.init_hidden(inp.size(1))
            log_pi, hidden = model_pitheta(inp, hidden, len_inp, mask_tar) # outpt [seq_len ,batch, ntok]
            log_pi_a = get_log_r(log_pi, action, mask_tar, ce_criterion).sum(0)
        else:
            log_pi_a = log_pi_a_q
        
        acceptance_rate += (rewards!=0).float().mean().cpu().numpy().item(0)
        tr_feats_pi = get_features(x, motifs, feat).mean(0).cpu().numpy()
        

        J_theta = -1*torch.mul(log_pi_a, rewards)
        
        loss = J_theta.mean()  
        if 'crit' in args.train2:
            loss += args.rl_value_loss_coeff*value_loss        

        total_loss += loss.data.float()
        T_loss += loss.data.float()
        if jj % log_interval == 0 and jj > 0:

            print('rewards', rewards.mean().cpu().numpy(), 'plambda', P_lambda.data.mean().cpu().numpy(), 'train ds feat = ', tr_feats_pi)
            idx = random.randrange(x.size(1))
            print(x[:,idx].cpu().numpy())


            cur_loss = total_loss / (log_interval)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | seq_len {} | acc_rate {:5.8f} | '
                  'pg loss {}'.format(
                epoch, jj, len(nbatches), 
                np.round(len_inp.float().mean().data.cpu().numpy(),decimals=1), 
                      acceptance_rate / log_interval , cur_loss))
            acceptance_rate = 0
            total_loss = 0
            start_time = time.time()

            model_q.train()
            model_pitheta.train()
            
        loss.backward()

        if jj % log_interval == 0 and jj > 0:
            if args.tensorboard:
                writer.add_scalar('pitheta/rewards', rewards.mean().data.cpu().numpy(), epoch*(len(nbatches))+jj)
                if 'crit' in args.train2:
                    writer.add_scalar('pitheta/critic', est_z.mean().data.cpu().numpy(), epoch*(len(nbatches))+jj)
                writer.add_scalar('pitheta/P_lambda', P_lambda.mean().data.cpu().numpy(), epoch*(len(nbatches))+jj)
                writer.add_scalar('pitheta/r(x)', torch.exp(log_r).mean().data.cpu().numpy(), epoch*(len(nbatches))+jj)
                writer.add_scalar('pitheta/seq_len', len_inp.float().mean().data.cpu().numpy(), epoch*(len(nbatches))+jj)
                writer.add_histogram('pitheta/tr_feats_sampled', tr_feats_pi, epoch*(len(nbatches))+jj) 
                for name, param in model_pitheta.named_parameters():
                    if name == 'encoder.weight': continue 
                    writer.add_histogram('pitheta/grad_'+name, param.grad.data.cpu().numpy(), epoch*(len(nbatches))+jj)
            if args.wandb:
                if 'crit' in args.train2:
                    wandb.log({'epoch': epoch*len(nbatches)+jj, 'rewards': returns.mean().data.cpu().numpy(), 
                    'pitheta_P_lambda': P_lambda.mean().data.cpu().numpy(), 'pitheta_r(x)': torch.exp(log_r).mean().data.cpu().numpy(),
                    'pitheta_seq_len': len_inp.float().mean().data.cpu().numpy(),  'pitheta_feats': tr_feats_pi, 'advantage':rewards.mean().data.cpu().numpy(), 'est_z':est_z.mean().data.cpu().numpy()})
                else:
                    wandb.log({'epoch': epoch*len(nbatches)+jj, 'rewards': rewards.mean().data.cpu().numpy(), 
                        'pitheta_P_lambda': P_lambda.mean().data.cpu().numpy(), 'pitheta_r(x)': torch.exp(log_r).mean().data.cpu().numpy(),
                        'pitheta_seq_len': len_inp.float().mean().data.cpu().numpy(),  'pitheta_feats': tr_feats_pi})
        # to prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm(model_pitheta.parameters(), clip)

        if args.optim == 'manual_lr':
            for n, p in model_pitheta.named_parameters():
                    if n == 'encoder.weight': continue
                    p.data.add_(-lr, p.grad.data)
        else:
            optimizer_pi.step()
    return model_pitheta, T_loss/len(nbatches), model_q




def ppo_update_one_epoch(model_pitheta, trajectories, ce_criterion, optimizer_pi, lr, epoch, writer):
    #batch_size = 4000
    batch_size = args.rl_mini_batch
    clip_param = args.rl_clip_param
    

    source_data = trajectories['x']

    batches_id = list(range(0, source_data.size(1), batch_size))
    shuffle(batches_id)

    all_idx = list(range(source_data.size(1)))
    shuffle(all_idx)
    trajectories['x'] = trajectories['x'][:,all_idx]
    trajectories['logpi_k_a'] = trajectories['logpi_k_a'][all_idx]
    if 'crit' in args.train2:
        trajectories['r'] = trajectories['r'][all_idx]
    trajectories['adv'] = trajectories['adv'][all_idx]

    T_loss = 0
    total_loss = 0
    approx_ent = 0
    approx_kl = 0

    for i, batch in enumerate(batches_id):
        # source_data[:,batch:batch+batch_size]
        x = trajectories['x'][:, batch:batch+batch_size]
        len_tar, mask_tar, data, action = get_length_mask(x)
        action = torch.mul(action.float(), mask_tar[:,:,0]).long()
        batch_size_i = mask_tar.size()[1]
        hidden = model_pitheta.init_hidden(x.size(1))
        model_pitheta.zero_grad()
        optimizer_pi.zero_grad()
        if 'crit' in args.train2:
            pi_output, hidden, est_z = model_pitheta(data, hidden, 0, mask_tar, critic=True)
        else:
            pi_output, hidden = model_pitheta(data, hidden, 0, mask_tar) # outpt [seq_len ,batch, ntok]

        log_pi = get_log_r(pi_output, action, mask_tar, ce_criterion).sum(0) #  [(n+1) x batch x 1] -> [batch x 1]
        logpi_k_a = trajectories['logpi_k_a'][batch:batch+batch_size]
        adv_targ = trajectories['adv'][batch:batch+batch_size]

        # PPO objectives
        ratio = torch.exp(log_pi - logpi_k_a)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
        loss = J_theta = -torch.min(surr1, surr2).mean()

        approx_kl += (- log_pi + logpi_k_a).mean().data.cpu().numpy().item(0)
        approx_ent += (- log_pi).mean().data.cpu().numpy().item(0)

        if 'crit' in args.train2:
            rewards = trajectories['r'][batch:batch+batch_size]
            value_loss = 0.5 * (rewards - est_z).pow(2).mean()
            loss += value_loss
      
        idx = random.randrange(x.size(1))

        total_loss += loss.data.float()
        T_loss += loss.data.float()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / (log_interval)
            print('| iter {:3d} | {:5d}/{:5d} batches | seq_len {:3.1f} | '
                  'pi_ppo loss {} | kl {} | ent {}'.format(
                epoch, i, len(batches_id), 
                np.round(len_tar.float().mean().data.cpu().numpy(),decimals=1), 
                      cur_loss,  approx_kl/(i+1), approx_ent/(i+1)))
            print(x[:,idx].cpu().numpy())        
            total_loss = 0
            
            
        loss.backward()
        if i % log_interval == 0 and args.tensorboard:
                for name, param in model_pitheta.named_parameters():
                    if name == 'encoder.weight': continue
                    writer.add_histogram('pitheta/grad_'+name, param.grad.data.cpu().numpy(), epoch*len(batches_id)+i)
        #writer.add_histogram('data/rewards', rewards.clone().cpu().data.numpy(), epoch*(len(nbatches))+jj)
        #writer.add_histogram('data/r(x)', torch.exp(log_r).clone().cpu().data.numpy(), epoch*(len(nbatches))+jj)
        #writer.add_scalar('data/train_J_theta', loss, epoch*(len(nbatches))+jj)
        #writer.add_scalar('data/seq_len', len_inp.float().mean().data, epoch*(len(nbatches))+jj)
        # to prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm(model_pitheta.parameters(), clip)

        if args.optim == 'manual_lr':
            for n, p in model_pitheta.named_parameters():
                    if n == 'encoder.weight': continue
                    p.data.add_(-lr, p.grad.data)
        else:
            optimizer_pi.step()

    return model_pitheta, approx_kl/len(batches_id), approx_ent/len(batches_id), T_loss/len(batches_id)

def ppo_update_one_epoch_flat(model_pitheta, trajectories, ce_criterion, optimizer_pi, lr, epoch, writer):
    #batch_size = 4000
    batch_size = args.rl_mini_batch
    clip_param = args.rl_clip_param
    

    source_data = trajectories['inp']

    batches_id = list(range(0, source_data.size(1), batch_size))
    shuffle(batches_id)

    all_idx = list(range(source_data.size(1)))
    shuffle(all_idx)
    #print(trajectories['tar'].size(1), trajectories['logpi_k_a'].size(0), trajectories['c_hid'].size(1), trajectories['r'].size(0))
    assert trajectories['tar'].size(1) == trajectories['logpi_k_a'].size(0) == trajectories['c_hid'].size(1) == trajectories['r'].size(0)
    trajectories['tar'] = trajectories['tar'][:,all_idx]
    trajectories['inp'] = trajectories['inp'][:,all_idx]
    trajectories['logpi_k_a'] = trajectories['logpi_k_a'][all_idx]
    trajectories['c_hid'] = trajectories['c_hid'][:,all_idx,:]
    trajectories['hid'] = trajectories['hid'][:,all_idx,:]
    if 'crit' in args.train2:
        trajectories['r'] = trajectories['r'][all_idx]
    trajectories['adv'] = trajectories['adv'][all_idx]

    T_loss = 0
    total_loss = 0
    approx_ent = 0
    approx_kl = 0

    for i, batch in enumerate(batches_id):
        
        data, action = trajectories['inp'][:,batch:batch+batch_size].cuda(), trajectories['tar'][:,batch:batch+batch_size].cuda()
        # [1 x batch x 1]
        mask_tar = (action != PAD).unsqueeze(2).float().cuda()
        action = torch.mul(action.float(), mask_tar[:,:,0]).long()
        #batch_size_i = mask_tar.size(1)
        
        model_pitheta.zero_grad()
        optimizer_pi.zero_grad()

        hidden = (trajectories['hid'][:, batch:batch+batch_size].contiguous().cuda(), trajectories['c_hid'][:, batch:batch+batch_size].contiguous().cuda())
        # outpt [ 1 x seq_len*batch x ntok]
        if 'crit' in args.train2:
            pi_output, hidden, est_z = model_pitheta(data, hidden, 0, mask_tar, critic=True)
        else:
            pi_output, hidden = model_pitheta(data, hidden, 0, mask_tar)

        log_pi = get_log_r(pi_output, action, mask_tar, ce_criterion)[0,:,:] # [1 x batch x 1]
        logpi_k_a = trajectories['logpi_k_a'][batch:batch+batch_size].cuda()
        adv_targ = trajectories['adv'][batch:batch+batch_size].cuda()

        # PPO objectives
        ratio = torch.exp(log_pi - logpi_k_a)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
        loss = J_theta = -torch.min(surr1, surr2).mean()

        approx_kl += (- log_pi + logpi_k_a).mean().data.cpu().numpy().item(0)
        approx_ent += (- log_pi).mean().data.cpu().numpy().item(0)

        if 'crit' in args.train2:
            rewards = trajectories['r'][batch:batch+batch_size].cuda()
            value_loss = 0.5 * (rewards - est_z).pow(2).mean()
            loss += value_loss

        
        #print('rewards', rewards.mean().cpu().numpy(), 'train ds feat = ', get_features(x, motifs, feat).mean(0).cpu().numpy())
        idx = random.randrange(data.size(1))

        total_loss += loss.data.float()
        T_loss += loss.data.float()
        if i % log_interval*10 == 0 and i > 0:
            cur_loss = total_loss / (log_interval)
            print('| iter {:3d} | {:5d}/{:5d} batches | '
                  'pi_ppo loss {} | kl {} '.format(
                epoch, i, len(batches_id), 
                cur_loss,  approx_kl/(i+1)))
            #print(data[:,idx].cpu().numpy())        
            total_loss = 0
            
            
        loss.backward()
        if  i % log_interval*10 == 0 and args.tensorboard:
                for name, param in model_pitheta.named_parameters():
                    if name == 'encoder.weight': continue
                    writer.add_histogram('pitheta/grad_'+name, param.grad.data.cpu().numpy(), epoch*len(batches_id)+i)
        #writer.add_histogram('data/rewards', rewards.clone().cpu().data.numpy(), epoch*(len(nbatches))+jj)
        #writer.add_histogram('data/r(x)', torch.exp(log_r).clone().cpu().data.numpy(), epoch*(len(nbatches))+jj)
        #writer.add_scalar('data/train_J_theta', loss, epoch*(len(nbatches))+jj)
        #writer.add_scalar('data/seq_len', len_inp.float().mean().data, epoch*(len(nbatches))+jj)
        # to prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm(model_pitheta.parameters(), clip)

        if args.optim == 'manual_lr':
            for n, p in model_pitheta.named_parameters():
                    if n == 'encoder.weight': continue
                    p.data.add_(-lr, p.grad.data)
        else:
            optimizer_pi.step()

    return model_pitheta, approx_kl/len(batches_id), approx_ent/len(batches_id), T_loss/len(batches_id)

# PPO for critic case:
# advantage = Q(s,a) - V(s)

def train_pi_theta_ppo(model_pitheta, model_plambda, ce_criterion, epoch, lr, motifs, feat, optimizer_pi, writer, z_estim_mc, model_q):
    # collect trajectories using current pi_theta_k
    # find a new pi_theta that maximizes the PPO objective on these trajectories
    # use approximate KL for early stopping 
    batch_size = 4000
    N = int((1.0*args.distill_size*args.rl_scale_iter)/batch_size)
    print('number of workers %d'%N)
    trajectories = {'r': torch.zeros((0,1)).cuda(), 
                    'adv': torch.zeros((0,1)).cuda(),
                    'x':  torch.zeros((args.n+2,0)).cuda().long(), 
                    'logpi_k_a': torch.zeros((0,1)).cuda(),
                    }
    epochs = 40
    target_kl = args.rl_target_kl

    if 'fix_length' in args.debug_opt:
            max_len = args.n+1
    else:
        max_len = 5*args.n

    pl_hidden = model_plambda.init_hidden(batch_size)
    for wrkr in range(N):
        # sample x ~ pi_theta(.)
        if 'crit' in args.train2:
            x, log_pi, inp, len_inp, action, mask_tar, est_z = sample_data_inp_targ_vary(model_pitheta, batch_size, max_len=max_len, critic=True)
            # get values for the last action
            # [seq_len x batch x 1] -> [batch x 1]
            est_z = torch.sum(est_z.squeeze().t().contiguous() * to_one_hot(len_inp-1, n_dims=est_z.size(0)), dim = 1).unsqueeze(1)
        else:
            x, log_pi, inp, len_inp, action, mask_tar = sample_data_inp_targ_vary(model_pitheta, batch_size, max_len=max_len)
        
        # get Plambda(x)
        r_output, _, log_lin = model_plambda(inp, pl_hidden, len_inp, mask_tar)
        if not 'wn' in args.train2:
            log_r = get_log_r(r_output, action, mask_tar, ce_criterion).sum(0) # [batch x 1]
        else:
            log_r = r_output
        P_lambda = torch.exp(log_r + log_lin)
        # [(n+1) x batch x 1]
        log_pi_a = get_log_r(log_pi, action, mask_tar, ce_criterion)
        log_pi_a = log_pi_a.sum(0)
        
        if 'dppo' in args.train2:
            rewards = torch.exp(log_r + log_lin - log_pi_a)/z_estim_mc
        elif 'ppo' in args.train2:
            rewards = P_lambda
            if not 'wn' in args.train2:
                rewards = rewards*(2**args.n)

        advantage = rewards.clone()

        if 'crit' in args.train2:
            assert rewards.size() == est_z.size()
            advantage = rewards - est_z

        print('rewards', rewards.mean().data.cpu().numpy(), 'P_lambda', P_lambda.mean().data.cpu().numpy())
        idx = random.randrange(x.size(1))
        print(x[:,idx].cpu().numpy())
        if args.wandb:
            if 'crit' in args.train2:
                wandb.log({'epoch': epoch*N+wrkr, 'rewards': rewards.mean().data.cpu().numpy(), 
                'pitheta_P_lambda': P_lambda.mean().data.cpu().numpy(),
                'pitheta_seq_len': len_inp.float().mean().data.cpu().numpy(), 'advantage':advantage.mean().data.cpu().numpy(), 'est_z':est_z.mean().data.cpu().numpy()})
            else:
                wandb.log({'epoch': epoch*N+wrkr, 'rewards': rewards.mean().data.cpu().numpy(), 
                'pitheta_P_lambda': P_lambda.mean().data.cpu().numpy(),
                'pitheta_seq_len': len_inp.float().mean().data.cpu().numpy()})
        if args.tensorboard:
            writer.add_scalar('pitheta/rewards', rewards.mean().data.cpu().numpy(), epoch*N+wrkr)
            writer.add_scalar('pitheta/P_lambda', P_lambda.mean().data.cpu().numpy(), epoch*N+wrkr)
            writer.add_scalar('pitheta/seq_len', len_inp.float().mean().data.cpu().numpy(), epoch*N+wrkr)
        # [seq_len x batch]
        #trajectories['a'] =  torch.cat(( trajectories['a'], action), dim=0)
        if 'crit' in args.train2:
            trajectories['r'] =  torch.cat(( trajectories['r'], rewards), dim=0).detach()
        trajectories['x'] = cat_variable_length(trajectories['x'], x).detach()
        # [batch x 1]
        trajectories['adv'] =  torch.cat(( trajectories['adv'], advantage), dim=0).detach()
        trajectories['logpi_k_a'] =  torch.cat(( trajectories['logpi_k_a'], log_pi_a), dim=0).detach()
            
    tr_feats = get_features(trajectories['x'], motifs, feat).mean(0).cpu().numpy()
    print('train ds feat = ', tr_feats)
    model_pitheta.train()
    total_loss = 0.
    start_time = time.time()
    if args.tensorboard:
        writer.add_histogram('pitheta/tr_feats_sampled', tr_feats, epoch)

    T_loss = 0

    for e in range(epochs):
                
        model_pitheta, approx_kl, approx_ent, loss = ppo_update_one_epoch(model_pitheta, trajectories, ce_criterion, optimizer_pi, lr, e, writer)

        total_loss += loss.data.float()
        T_loss += loss.data.float()
        if e % log_interval == 0 and e > 0:
            cur_loss = total_loss / (log_interval)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} iters | kl {} | ent {:5.8f} | '
                  'pg loss {}'.format(
                epoch, e, epochs, 
                approx_kl, approx_ent , cur_loss))
            total_loss = 0
            start_time = time.time()

        if args.wandb:
            wandb.log({'epoch': epoch*epochs + e, 'pitheta_ppo_loss': loss, 'pitheta_approx_kl': approx_kl})
        if args.tensorboard:
            writer.add_scalar('pitheta/approx_kl', approx_kl, epoch*epochs+e)
            writer.add_scalar('pitheta/ppo_loss',loss.mean().data.cpu().numpy(), epoch*epochs+e)
            
        if approx_kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%e)
                break


    return model_pitheta, T_loss/epochs, model_q

def train_pi_theta_ppo_flat(model_pitheta, model_plambda, ce_criterion, epoch, lr, motifs, feat, optimizer_pi, writer, z_estim_mc, model_q):
    # collect trajectories using current pi_theta_k for every action individually
    # find a new pi_theta that maximizes the PPO objective on these trajectories
    # use approximate KL for early stopping 
    batch_size = 4000
    N = 20
    trajectories = {'r': torch.zeros((0,1)),
                    'adv': torch.zeros((0,1)), 
                    'inp':  torch.zeros((1,0)).long(), 
                    'tar':  torch.zeros((1,0)).long(),
                    'logpi_k_a': torch.zeros((0,1)),
                    'hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid),
                    'c_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid),
                    }
    epochs = 40
    target_kl = args.rl_target_kl
    all_x = torch.zeros((args.n+2,0)).cuda().long()


    if 'fix_length' in args.debug_opt:
            max_len = args.n+1
    else:
        max_len = 3*args.n

    pl_hidden = model_plambda.init_hidden(batch_size)
    for wrkr in range(N):
        # sample batch of x ~ pi_theta(.)
        if 'crit' in args.train2:
            x, log_pi, inp, len_inp, action, mask_tar, hids, c_hids, est_z = sample_data_inp_targ_vary_hid(model_pitheta, batch_size, max_len=max_len, critic=True)
            # est_z: [seq_len x batch x 1]  -> []
            est_z = est_z.view(-1, 1)
        else:
            x, log_pi, inp, len_inp, action, mask_tar, hids, c_hids = sample_data_inp_targ_vary_hid(model_pitheta, batch_size, max_len=max_len)
        
        # get Plambda(x)
        r_output, _, log_lin = model_plambda(inp, pl_hidden, len_inp, mask_tar)
        if not 'wn' in args.train2:
            log_r = get_log_r(r_output, action, mask_tar, ce_criterion).sum(0) # [batch x 1]
        else:
            log_r = r_output
        P_lambda = torch.exp(log_r + log_lin)
        # [(n+1) x batch x 1]
        #print('x', x.size(), 'log_pi', log_pi.size(), 'a', action.size())
        log_pi_a_all = get_log_r(log_pi, action, mask_tar, ce_criterion)
        log_pi_a = log_pi_a_all.sum(0)
        
        
        if 'dppo' in args.train2:
            if 'crit' in args.train2:
                # [batch x 1] -> [(n+1) x batch]
                log_P_lambda = (log_r + log_lin).repeat(1, inp.size(0)).t().contiguous()
                assert log_P_lambda.size() == log_pi_a_all.squeeze().size()
                # unbiased estimate for Z(s) = P_lambda(x)/pi_theta(x|s)
                log_pi_x_s = log_pi_a_all.squeeze().clone()
                for i in range(log_pi_x_s.size(0)):
                    for j in range(log_pi_x_s.size(1)):
                        log_pi_x_s[i,j] = log_pi_a_all[i:,j].sum()
                rewards = torch.exp(log_P_lambda - log_pi_x_s).view(-1, 1)
                advantage = rewards - est_z
                assert rewards.size() == est_z.size()
            else:
                rewards = torch.exp(log_r + log_lin - log_pi_a)
        elif 'ppo' in args.train2:           
            if 'crit' in args.train2:
                rewards = P_lambda.repeat(1, inp.size(0)).view(-1,1)
                advantage = rewards - est_z
            else:
                rewards = P_lambda
                if not 'wn' in args.train2:
                    rewards = rewards*(2**args.n)

        if not 'crit' in args.train2:
            # [batch x 1] -> [batch x (n+1)] -> [(n+1) * batch x 1]
            advantage = rewards = rewards.repeat(1, inp.size(0)).t().contiguous().view(-1,1)
            

        print('rewards', rewards.mean().data.cpu().numpy(), 'P_lambda', P_lambda.mean().data.cpu().numpy())
        idx = random.randrange(x.size(1))
        print(x[:,idx].cpu().numpy())
        # [ batch*seq_len x 1]
        log_pi_a_all = log_pi_a_all.view(-1, 1)
        inp = inp.view(1, -1)
        action = action.view(1, -1)

        if args.wandb:
            if 'crit' in args.train2:
                wandb.log({'epoch': epoch*N+wrkr, 'rewards': rewards.mean().data.cpu().numpy(), 
                'pitheta_P_lambda': P_lambda.mean().data.cpu().numpy(),
                'pitheta_seq_len': len_inp.float().mean().data.cpu().numpy(), 'advantage':advantage.mean().data.cpu().numpy(), 'est_z':est_z.mean().data.cpu().numpy()})
            else:
                wandb.log({'epoch': epoch*N+wrkr, 'rewards': rewards.mean().data.cpu().numpy(), 
                'pitheta_P_lambda': P_lambda.mean().data.cpu().numpy(),
                'pitheta_seq_len': len_inp.float().mean().data.cpu().numpy()})
        if args.tensorboard:
            writer.add_scalar('pitheta/rewards', rewards.mean().data.cpu().numpy(), epoch*N+wrkr)
            writer.add_scalar('pitheta/P_lambda', P_lambda.mean().data.cpu().numpy(), epoch*N+wrkr)
            writer.add_scalar('pitheta/seq_len', len_inp.float().mean().data.cpu().numpy(), epoch*N+wrkr)

        all_x = cat_variable_length(all_x, x).detach()
        # due to the small GPU memory move trajectories to cpu
        trajectories['c_hid'] =  torch.cat((trajectories['c_hid'], c_hids), dim=1).detach()
        trajectories['hid'] =  torch.cat((trajectories['hid'], hids), dim=1).detach()
        trajectories['inp'] =  torch.cat((trajectories['inp'], inp.cpu()), dim=1).detach()
        trajectories['tar'] =  torch.cat((trajectories['tar'], action.cpu()), dim=1).detach()
        # [batch x 1]
        if 'crit' in args.train2:
            trajectories['r'] =  torch.cat(( trajectories['r'], rewards.cpu()), dim=0).detach()
        trajectories['adv'] =  torch.cat(( trajectories['adv'], advantage.cpu()), dim=0).detach()
        trajectories['logpi_k_a'] =  torch.cat(( trajectories['logpi_k_a'], log_pi_a_all.cpu()), dim=0).detach()
            
    tr_feats_pi = get_features(all_x, motifs, feat).mean(0).cpu().numpy()
    print('train ds feat = ', tr_feats_pi)
    if args.wandb:
        wandb.log({'epoch': epoch, 'pitheta_feats': tr_feats_pi})
    if args.tensorboard:
        writer.add_histogram('pitheta/tr_feats_sampled', tr_feats_pi, epoch)

    model_pitheta.train()
    total_loss = 0.
    start_time = time.time()

    T_loss = 0

    for e in range(epochs):
                
        model_pitheta, approx_kl, approx_ent, loss = ppo_update_one_epoch_flat(model_pitheta, trajectories, ce_criterion, optimizer_pi, lr, e, writer)

        total_loss += loss.data.float()
        T_loss += loss.data.float()
        if args.wandb:
            wandb.log({'epoch': epoch*epochs + e, 'pitheta_ppo_loss': loss, 'pitheta_approx_kl': approx_kl})
        if args.tensorboard:
            writer.add_scalar('pitheta/approx_kl', approx_kl, epoch*epochs+e)
            writer.add_scalar('pitheta/ppo_loss',loss.mean().data.cpu().numpy(), epoch*epochs+e)

        if e % log_interval == 0 and e > 0:
            cur_loss = total_loss / (log_interval)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} iters | kl {} | ent {:5.8f} | '
                  'pg loss {}'.format(
                epoch, e, epochs, 
                approx_kl, approx_ent , cur_loss))
            total_loss = 0
            start_time = time.time()
            
        if approx_kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%e)
                break

    trajectories = None
    torch.cuda.empty_cache()
    return model_pitheta, T_loss/epochs, model_q




def ac_dpg_update_one_epoch(model_pitheta, trajectories, ce_criterion, optimizers, lr, epoch, writer, model_crit, Epoch):
    #batch_size = 4000
    model_pitheta.train()
    model_crit.train()

    batch_size = args.rl_mini_batch
    clip_param = args.rl_clip_param
    
    [optimizer_pi, optimizer_w] = optimizers

    source_data = trajectories['inp']

    batches_id = list(range(0, source_data.size(1), batch_size))
    shuffle(batches_id)

    all_idx = list(range(source_data.size(1)))
    shuffle(all_idx)
    #print(trajectories['tar'].size(),  trajectories['c_hid'].size(), trajectories['r'].size())
    assert trajectories['tar'].size(1)  == trajectories['q_c_hid'].size(1) == trajectories['r'].size(0)
    trajectories['tar'] = trajectories['tar'][:,all_idx]
    trajectories['inp'] = trajectories['inp'][:,all_idx]
    trajectories['q_c_hid'] = trajectories['q_c_hid'][:,all_idx,:]
    trajectories['q_hid'] = trajectories['q_hid'][:,all_idx,:]
    trajectories['cr_c_hid'] = trajectories['cr_c_hid'][:,all_idx,:]
    trajectories['cr_hid'] = trajectories['cr_hid'][:,all_idx,:]
    trajectories['r'] = trajectories['r'][all_idx]
    trajectories['adv'] = trajectories['adv'][all_idx]

    T_loss, V_loss = 0, 0
    total_loss = 0
    approx_ent = 0
    approx_kl = 0

    for i, batch in enumerate(batches_id):
        
        data, action = trajectories['inp'][:,batch:batch+batch_size].cuda(), trajectories['tar'][:,batch:batch+batch_size].cuda()
        # [1 x batch x 1]
        mask_tar = (action != PAD).unsqueeze(2).float().cuda()
        action = torch.mul(action.float(), mask_tar[:,:,0]).long()
        #batch_size_i = mask_tar.size(1)
        
        model_pitheta.zero_grad()
        optimizer_pi.zero_grad()
        model_crit.zero_grad()
        optimizer_w.zero_grad()

        hidden = (trajectories['q_hid'][:, batch:batch+batch_size].contiguous().cuda(), trajectories['q_c_hid'][:, batch:batch+batch_size].contiguous().cuda())
        # outpt: [ 1 x batch x ntok]
        pi_output, _, _ = model_pitheta(data, hidden, 0, mask_tar, critic=True)

        hidden = (trajectories['cr_hid'][:, batch:batch+batch_size].contiguous().cuda(), trajectories['cr_c_hid'][:, batch:batch+batch_size].contiguous().cuda())
        # est_z: [1 x batch x ntok]
        _, _, est_z = model_crit(data, hidden, 0, mask_tar, critic=True)

        log_pi = get_log_r(pi_output, action, mask_tar, ce_criterion)[0,:,:] # [batch x 1]
        adv_targ = trajectories['adv'][batch:batch+batch_size].cuda()

        # AC D-PG objectives
        loss = J_theta = -1*torch.mul(log_pi, adv_targ).mean()
        
        rewards = trajectories['r'][batch:batch+batch_size].cuda()
         
        if 'ac_dpg_a' in args.train2:
            # add leaves to get value at the parent node
            est_z_s = est_z.sum(2).squeeze(0)
        else:
            # no leaves, parent node in the log domain
            est_z_s = est_z.squeeze()

        value_loss = args.rl_value_loss_coeff * (rewards.squeeze() - est_z_s).pow(2).mean()

        
        idx = random.randrange(data.size(1))

        total_loss += loss.data.float()
        T_loss += loss.data.float()
        V_loss += value_loss.data.float()
        if i % log_interval*10 == 0 and i > 0:
            cur_loss = total_loss / (log_interval)
            print('| iter {:3d} | {:5d}/{:5d} batches | '
                  'policy loss {}  '.format(
                epoch, i, len(batches_id), 
                cur_loss))
            #print(data[:,idx].cpu().numpy())        
            total_loss = 0
                      
        loss.backward()
        value_loss.backward()


        if  i % log_interval*10 == 0 and args.tensorboard:
                for name, param in model_pitheta.named_parameters():
                    if name == 'encoder.weight': continue
                    writer.add_histogram('pitheta/grad_'+name, param.grad.data.cpu().numpy(), epoch*len(batches_id)+i)
        # to prevent the exploding gradient problem
        if Epoch>1:  
            torch.nn.utils.clip_grad_norm(model_pitheta.parameters(), clip)
        torch.nn.utils.clip_grad_norm(model_crit.parameters(), clip)

        if args.optim == 'manual_lr':
            for n, p in model_pitheta.named_parameters():
                    if n == 'encoder.weight': continue
                    p.data.add_(-lr, p.grad.data)
            for n, p in model_crit.named_parameters():
                    if n == 'encoder.weight': continue
                    p.data.add_(-lr, p.grad.data)
        else:
            if Epoch>1:  
                optimizer_pi.step()
            optimizer_w.step()

    return model_pitheta, T_loss/len(batches_id), V_loss/len(batches_id), model_crit


def plan_z_estim_ac_dpg(model_q, x, model_crit):
    # output Z_planned: [(n+2) x batch]


    action = x[1:]

    batch_size_i = x.size(1)
    seq_len = x.size(0) 
    
    hids =  [0]*seq_len
    c_hids = [0]*seq_len
    
    hidden = model_crit.init_hidden(batch_size_i)   
    plan_z = torch.zeros((seq_len, batch_size_i)).cuda()
    END = 2
    inv_seq_mask = (x == END) + (x == PAD) # 0: sequence ; 1: finished sequence
    term_elem_mask = (x != END) # 0: finished sequence
    action =  torch.mul(action, (action != PAD).long())

    out = torch.zeros(seq_len, batch_size_i).cuda().long()
    len_inp = torch.ones((batch_size_i), dtype=torch.int64)

    for d in range(args.rl_plan_depth+1):

        for pos in range(seq_len):        

            symb = x[pos:pos+1]
            mask = (symb!=END).float().unsqueeze(2)
            
            if d == 0:
                # left cell
                if pos>0:
                    hidden = (hids[pos-1].cuda(), c_hids[pos-1].cuda())
            else:
                # bottom cell from previous layer
                hidden = (hids[pos].cuda(), c_hids[pos].cuda())
            # log domain est_z: [1 x batch x ninp]
            _, hidden, est_z = model_crit(symb, hidden, len_inp, mask, critic=True)


            hids[pos] =  hidden[0].detach().cpu()
            c_hids[pos] =  hidden[1].detach().cpu()

            if d < args.rl_plan_depth:

                if d ==0 and (pos < seq_len - 1):
                    # choose actions sampled from q
                    max_z_ind = action[pos]
                    #print('z', est_z.size(), 'a', action.size(), ninp, pos, batch_size_i)
                    max_z_val =  torch.sum(est_z.view(-1, ninp) * to_one_hot(action[pos], n_dims=ninp), dim = 1).view(batch_size_i, 1)
                else:
                    # choose best action according to Z estimates
                    max_z_val, max_z_ind = est_z.max(2)
                
                # accumulate Z value of the leaves that are not going to be expanded
                # mask in logsumexp maximum elements for Z
                not_expand_est_z = torch.where(to_one_hot(max_z_ind, n_dims=ninp).view(est_z.size()).byte(), (torch.ones(est_z.size())*float('-inf')).cuda(), est_z )
                #plan_z[pos] = plan_z[pos] + torch.mul(est_z.sum(2).view(batch_size_i) - max_z_val.view(batch_size_i), torch.mul((inv_seq_mask[pos] == 0).float(), (symb != END).float().squeeze()))
                plan_z[pos] = plan_z[pos] + torch.mul(logsumexp(not_expand_est_z, dim=2).view(batch_size_i), torch.mul((inv_seq_mask[pos] == 0).float(), (symb != END).float().squeeze()))

                # for the terminal elements add the final Z value
                plan_z[pos] = plan_z[pos] + torch.mul(logsumexp(est_z, dim=2).view(batch_size_i), torch.mul((term_elem_mask[pos] == 0).float(), (symb == END).float().squeeze()))


                # for accumulated leaves
                term_elem_mask[pos] = term_elem_mask[pos] + (symb == END)
                # for finished sequences or PAD tokens mask > 0
                inv_seq_mask[pos] = inv_seq_mask[pos] + (max_z_ind == END)
                out[pos] = max_z_ind.view(batch_size_i)

            else:
                # for unfinished sequences add all leaves
                plan_z[pos] = plan_z[pos] + torch.mul(logsumexp(est_z, dim=2).view(batch_size_i), (term_elem_mask[pos] == 0).float())

        x = out
    del hids
    del c_hids
    del inv_seq_mask

    return plan_z

def train_pi_theta_ac_dpg(model_pitheta, model_plambda, ce_criterion, epoch, lr, motifs, feat, optimizers, writer, z_estim_mc, model_q, model_crit):
    # collect trajectories using q
    # find a new pi_theta that maximizes the AC D-PG objective on these trajectories

    batch_size = 4000
    N = 20
    trajectories = {'r': torch.zeros((0,1)), 'adv': torch.zeros((0,1)), 'inp':  torch.zeros((1,0)).long(), 'tar':  torch.zeros((1,0)).long(),
                    'logpi_k_a': torch.zeros((0,1)), 'q_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid), 'q_c_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid),
                    'cr_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid), 'cr_c_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid)
                    }
    
    target_kl = args.rl_target_kl
    all_x = torch.zeros((args.n+2,0)).long()


    if 'fix_length' in args.debug_opt:
            max_len = args.n+1
    else:
        max_len = 3*args.n

    pl_hidden = model_plambda.init_hidden(batch_size)
    if not 'stable_q' in args.train2:
            model_q = model_pitheta


    for wrkr in range(N):
        # get samples from q
        x, log_pi, inp, len_inp, action, mask_tar, hids, c_hids = sample_data_inp_targ_vary_hid(model_q, batch_size, max_len=max_len)
        
        # get Plambda(x)
        r_output, _, log_lin = model_plambda(inp, pl_hidden, len_inp, mask_tar)
        if not 'wn' in args.train2:
            log_r = get_log_r(r_output, action, mask_tar, ce_criterion).sum(0) # [batch x 1]
        else:
            log_r = r_output
        P_lambda = torch.exp(log_r + log_lin)
        # [(n+1) x batch x 1]
        log_pi_a_all = get_log_r(log_pi, action, mask_tar, ce_criterion)
        
        # [batch x 1] -> [batch x (n+1)] -> [(n+1) x batch]
        log_P_lambda = (log_r + log_lin).repeat(1, inp.size(0)).t().contiguous()
        #assert log_P_lambda.size() == log_pi_a_all.squeeze().size()
        log_pi_x_s = log_pi_a_all.squeeze().clone()
        for i in range(log_pi_x_s.size(0)):
            for j in range(log_pi_x_s.size(1)):
                log_pi_x_s[i,j] = log_pi_a_all[i:,j].sum()

        # unbiased estimate for log_Z(s) = log(P_lambda(x)/pi_theta(x|s))
        rewards = (log_P_lambda - log_pi_x_s).view(-1, 1)
        

        len_inp = (x!= PAD).sum(0)
        mask_tar = (x != PAD).unsqueeze(2).float().cuda()        
        
        if 'ac_dpg_a' in args.train2: 
            # log domain
            est_z = plan_z_estim_ac_dpg(model_q, x, model_crit)

        else:
            hidden = model_crit.init_hidden(x.size(1))
            mask = torch.ones((1, x.size(1), 1)).cuda()
            cr_hids =  torch.zeros(model_crit.nlayers, x.size(1)*x.size(0), model_crit.nhid)
            cr_c_hids = torch.zeros(model_crit.nlayers, x.size(1)*x.size(0), model_crit.nhid)
            est_z = torch.zeros(x.size(0), x.size(1), 1).cuda()
            # est_z: [seq_len x batch x 1] -- log domain
            for i in range(x.size(0)):
                # [1 x batch x ntok]
                _, hidden, est_z_i = model_crit(x[i:i+1], hidden, len_inp, mask, critic=True)
                cr_hids[:,i*x.size(1):(i+1)*x.size(1),:,] =  hidden[0].cpu().detach()
                cr_c_hids[:,i*x.size(1):(i+1)*x.size(1),:,] = hidden[1].cpu().detach()
                est_z[i:i+1] = est_z_i
            #_, _, est_z = model_crit(x, hidden, len_inp, mask_tar, critic=True) # outpt [n+2, batch, ntok]
           
        est_z_now = est_z[:-1].view(-1, 1)
        est_z_next = est_z[1:].view(-1, 1)

        log_pi_a_s = log_pi_a_all.view(-1, 1)

        ratio_zs = torch.clamp(est_z_next - est_z_now, max=0)
        advantage = torch.exp(ratio_zs - log_pi_a_s)

        # ------------------ inspect max advantages ---------------------

        max_adv, max_idx = torch.max(advantage, 0)

        #print(max_idx, log_pi_a_s.size(), advantage.size(), est_z_now.size(), est_z.size(), P_lambda.size())
        
        if args.wandb:
            wandb.log({'epoch': epoch*N+wrkr, 'max_adv': max_adv.mean().data.cpu().numpy(),
                'max_q_adv': torch.exp(log_pi_a_s[max_idx, 0]).data.cpu().numpy(), 
                'max_z_now': est_z_now[max_idx, 0].data.cpu().numpy(), 'max_z_next': est_z_next[max_idx, 0].data.cpu().numpy(),
                'est_z_5':est_z[5].mean().data.cpu().numpy()})
        print('max_adv',max_adv.mean().data.cpu().numpy(),
                'max_q_adv', torch.exp(log_pi_a_s[max_idx, 0]).data.cpu().numpy())

        # ------------------------------------------------------------

        len_inp = len_inp - 1
        assert rewards.size() == est_z_now.size() == log_pi_a_s.size()
            
        print('rewards', rewards.mean().data.cpu().numpy(), 'P_lambda', P_lambda.mean().data.cpu().numpy())
        idx = random.randrange(x.size(1))
        print(x[:,idx].cpu().numpy())
        # [ batch*seq_len x 1]
        log_pi_a_all = log_pi_a_all.view(-1, 1)
        inp = inp.view(1, -1)
        action = action.view(1, -1)


        if args.wandb:
            wandb.log({'epoch': epoch*N+wrkr, 'rewards': torch.exp(rewards).mean().data.cpu().numpy(), 
                'pitheta_P_lambda': P_lambda.mean().data.cpu().numpy(),
                'pitheta_seq_len': len_inp.float().mean().data.cpu().numpy(), 'advantage':advantage.mean().data.cpu().numpy(), 
                'est_z_0':est_z[0].mean().data.cpu().numpy()})
        if args.tensorboard:
            writer.add_scalar('pitheta/rewards', rewards.mean().data.cpu().numpy(), epoch*N+wrkr)
            writer.add_scalar('pitheta/P_lambda', P_lambda.mean().data.cpu().numpy(), epoch*N+wrkr)
            writer.add_scalar('pitheta/seq_len', len_inp.float().mean().data.cpu().numpy(), epoch*N+wrkr)

        all_x = cat_variable_length(all_x, x.cpu().detach())

        # due to the small GPU memory move trajectories to cpu
        trajectories['q_c_hid'] =  torch.cat((trajectories['q_c_hid'], c_hids), dim=1).detach()
        trajectories['q_hid'] =  torch.cat((trajectories['q_hid'], hids), dim=1).detach()
        trajectories['cr_c_hid'] =  torch.cat((trajectories['cr_c_hid'], cr_c_hids), dim=1).detach()
        trajectories['cr_hid'] =  torch.cat((trajectories['cr_hid'], cr_hids), dim=1).detach()
        trajectories['inp'] =  torch.cat((trajectories['inp'], inp.cpu()), dim=1).detach()
        trajectories['tar'] =  torch.cat((trajectories['tar'], action.cpu()), dim=1).detach()
        # [batch x 1]
        trajectories['r'] =  torch.cat(( trajectories['r'], rewards.cpu()), dim=0).detach()
        trajectories['adv'] =  torch.cat(( trajectories['adv'], advantage.cpu()), dim=0).detach()
        trajectories['logpi_k_a'] =  torch.cat(( trajectories['logpi_k_a'], log_pi_a_all.cpu()), dim=0).detach()

        if wrkr % 4 == 0 or (wrkr == N-1):
            e = wrkr
            model_pitheta, loss, v_loss, model_crit = ac_dpg_update_one_epoch(model_pitheta, trajectories, ce_criterion, optimizers, lr, e, writer, model_crit, epoch)

            if args.wandb:
                wandb.log({'epoch': epoch*N+wrkr, 'pitheta_policy_loss': loss, 'pitheta_crit_loss': v_loss})

            del trajectories
            trajectories = {'r': torch.zeros((0,1)), 'adv': torch.zeros((0,1)), 'inp':  torch.zeros((1,0)).long(), 'tar':  torch.zeros((1,0)).long(),
                    'logpi_k_a': torch.zeros((0,1)), 'q_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid), 'q_c_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid),
                    'cr_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid), 'cr_c_hid':  torch.zeros(model_pitheta.nlayers, 0, model_pitheta.nhid)
                    }
            
    tr_feats_pi = get_features(all_x, motifs, feat).mean(0).cpu().numpy()
    print('train ds feat = ', tr_feats_pi)
    if args.wandb:
        wandb.log({'epoch': epoch, 'pitheta_feats': tr_feats_pi})
    if args.tensorboard:
        writer.add_histogram('pitheta/tr_feats_sampled', tr_feats_pi, epoch)

    model_pitheta.train()

    #e = 0
    #model_pitheta, loss, model_crit = ac_dpg_update_one_epoch(model_pitheta, trajectories, ce_criterion, optimizers, lr, e, writer, model_crit)

    trajectories = None
    torch.cuda.empty_cache()
    return model_pitheta, loss, model_q, model_crit








def main():  
    # training-1: train proposal r on D and obtain P_lambda
    model_r, model_plambda, test_ce_r, test_ce_pl,theor_ent,tstamp,lambd, Epoch_start_time, writer, optimizer_r, all_data = training_1()

    # training-2: get pi_theta from P_lambda
    if 'cyclic' in args.train2:
        # cyclic improvement of pi -> r
        print('CYCLIC MODE')
        test_ce_pi,mfeat_pl,tstamp,Final_duration,train_l1_pl, model_pitheta, lambd, test_ce_pl = cyclic_r_plambda_pitheta(model_plambda, model_r, tstamp, Epoch_start_time, writer, optimizer_r, all_data)
    elif 'distill' in args.train2:
        #distill in one cyclic iteration
        print('distill in one cyclic iteration')
        test_ce_pi,mfeat_pl,tstamp,Final_duration,train_l1_pl, model_pitheta = r_plambda_distill_pitheta(model_plambda, model_r, tstamp, Epoch_start_time, writer, all_data)
    elif 'pg' in args.train2 or 'ppo' in args.train2:
        print('RL MODE')
        test_ce_pi,mfeat_pl,tstamp,Final_duration,train_l1_pl, model_pitheta = rl_pitheta(model_plambda, model_r, tstamp, Epoch_start_time, writer, all_data)



    # ------------------------------------------ frequency of motifs in samples from pi_theta and r ------------------------------------

    log_dir = os.path.join(args.logdir,'pg_methods/runs/chpt_%s'%(timestamp))   
    model_r = torch.load(os.path.join(log_dir,'chpt_%s_r.pt'%(timestamp)))

    motif_freq, avg_len = sample_from_rnn(model_pitheta)
    motif_freq_r, avg_len_r = sample_from_rnn(model_r)
    print('r avg_len', avg_len_r, 'r motif_freq', motif_freq_r)
    print('pi avg_len', avg_len, 'pi motif_freq', motif_freq)

    if args.wandb:
        wandb.log({'T2_duration': Final_duration, 'lambd':lambd, 'mfeat_pl':mfeat_pl, 'pitheta_mfreq': motif_freq, 'r_mfreq': motif_freq_r, 'pi_avg_len':avg_len, 
                    'r_avg_len':avg_len_r })

    tstamp = tstamp+'_'+str(args.rl_seed)+'_'+str(motif_freq)+'_'+str(avg_len) +'_'+str(motif_freq_r)+'_'+str(avg_len_r)    

    return [test_ce_r,test_ce_pi,test_ce_pl,train_l1_pl,theor_ent,tstamp,lambd,mfeat_pl,Final_duration]




if __name__ == "__main__":
    info_p_lambda = main()
    train = args.train+'_'+args.train2+'_'+args.debug_opt
    print(tuple(info_p_lambda+[args.mtype+'.'+all_motifs[args.n],train,args.feat,args.n,args.ds_size,args.job]))

    # ----------------------- store the results into database ---------------------------------
    if not args.test_run:
        # r_plambda_distill_pitheta.db
        with sqlite3.connect('/tmp-network/user/tparshak/r_plambda_pitheta.db', timeout=10) as conn:
            # this will be executed once because of the "IF NOT EXISTS" clause
            conn.execute('CREATE TABLE IF NOT EXISTS results (test_ce_r REAL,test_ce_pi REAL,test_ce_pl REAL,train_l1_pl REAL,theor_ent REAL,tstamp TEXT,lambd TEXT,mfeat_pl TEXT,plambda_time REAL,motif TEXT,train_reg TEXT,feat TEXT,n INTEGER,ds_size INTEGER,job INTEGER)')
            conn.execute('INSERT INTO results (test_ce_r,test_ce_pi,test_ce_pl,train_l1_pl,theor_ent,tstamp,lambd,mfeat_pl,plambda_time,motif,train_reg,feat,n,ds_size,job) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                tuple(info_p_lambda+[args.mtype+'.'+all_motifs[args.n],train,args.feat,args.n,args.ds_size,args.job]))


