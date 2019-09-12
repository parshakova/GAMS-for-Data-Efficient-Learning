
# coding: utf-8

# Allowing log domain weights
# 27.03.2019

# # WFSA operations and data generation with motifs

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import sys
from pprint import pprint
import math
from math import log, log2
import numpy as np
import copy
from scipy.special import logsumexp

import os
import argparse
from argparse import Namespace


# from IPython.core.display import Image, display, HTML


def lsr_sum(lst):
    return logsumexp(lst)
def lsr_prod(lst):
    return sum(lst)
def lsr_div(num,denum):
    return num-denum

def log_extended(x):
    if x == 0:
        return float('-inf')
    else:
        return log(x)

def gconvert(standard_real):
    if not LSR:
        return standard_real
    else:
        return log_extended(standard_real)

def convert_to_standard_real(x):
    if not LSR:
        return x
    else:
        # print(f'xxxxxxx: {x}')
        # set_trace()
        return math.exp(x)

def gsum(*args):
    '''
    gsum: generic sum, which can do LSR sum or standard sum
    '''
    if LSR:
        specific_sum = lsr_sum
    else:
        specific_sum = sum
    if len(args) == 1:     # then the first (and only) arg should be a list
        return specific_sum(args[0])
    else:                  # otherwise take the list of arguments and compute the sum on this list
        return specific_sum(args)

def myprod(lst):
    prod = 1.0
    for x in lst:
        prod *= x
    return prod

def gprod(*args):
    '''
    gprod: generic prod, which can do LSR prod or standard prod 
    '''
    if LSR:
        specific_prod = lsr_prod
    else:
        specific_prod = myprod
    if len(args) == 1:     # then the first (and only) arg should be a list
        return specific_prod(args[0])
    else:                  # otherwise take the list of arguments and compute the sum on this list
        return specific_prod(args)



def mydiv(num,denum):
    return num/denum

def gdiv(num,denum):
    '''
    gdiv: generic div, which can do LSR div or standard div
    exactly two arguments in this case
    '''
    if LSR:
        specific_div = lsr_div
    else:
        specific_div = mydiv # np.divide
    return specific_div(num,denum)


# # WFSA

# In[4]:


class WFSA:
    """
    WFSA class
    """

    # def __init__(self, initial_state=None, transitions=set(), prune_unreachables=False, prune_small_weight_states=False):
    def __init__(self, initial_state=None, transitions=set(), prune_unreachables=True, prune_small_weight_states=True):
        self.initial_state=initial_state
        
        if prune_small_weight_states:
            intermediary = WFSA(initial_state=initial_state,transitions=transitions,prune_unreachables=True,prune_small_weight_states=False)
            # intermediary.display(verbosity=1)
            transitions = intermediary.prune_small_weights_transitions()
            transitions = intermediary.prune_unreachable_states(initial_state,transitions)
            prune_unreachables = True
        if prune_unreachables:
            transitions=self.prune_unreachable_states(initial_state,transitions)
            
        self.transitions=transitions
        self.states = {initial_state} | {e[0] for e in transitions}
        self.transitions_dict = {}
        # self.final_states_dict = {e[0]:gconvert(e[1]) for e in self.transitions if len(e)==2}
        self.final_states_dict = {e[0]:e[1] for e in self.transitions if len(e)==2}
        self.final_states = set(self.final_states_dict.keys())
        self.vocab = {e[2] for e in self.transitions if len(e)==4}
        
        for transition in transitions:
            source = transition[0]
            value = self.transitions_dict.setdefault(source,set())
            if len(transition) == 4: # standard transition
                target = transition[1]
                label = transition[2]
                weight = transition[3]
                value.add((target,label,weight))
            if len(transition) == 2: # final state
                weight = transition[1]
                value.add((weight,))

    @classmethod
    def fromfilename(cls, filename):
        split_lines = [line.split() for line in open(filename,'r').readlines()]
        transitions=set()
        final_states_weights=set()
        first_line = True
        for sl in split_lines:
            if len(sl) == 4:
                source = sl[0]
                target = sl[1]
                label = sl[2]
                weight = gconvert(float(sl[3]))
                if source.isdigit():
                    source = int(source)
                if target.isdigit():
                    target=int(target)
                transitions.add((source,target,label,weight))
                if first_line:
                    initial_state=source
                    first_line=False
            if len(sl)==2: # final_state with weight
                source = sl[0]
                if source.isdigit():
                    source=int(source)
                weight = gconvert(float(sl[1]))
                transitions.add((source,weight))
                
        return cls(initial_state=initial_state,transitions=transitions)
    
    def display(self,verbosity=0):
        if verbosity == 1: # verbose
            pprint(vars(self))
        if verbosity == 0:
            pprint({'transitions':self.transitions})

    def is_final_state(self,state):
        return (state in self.final_states)

    def prune_unreachable_states(self,initial_state,transitions):
        '''eliminate states that are not connected to the initial state or to some final state'''
        connected_to_initial = {initial_state}
        connected_to_final = {t[0] for t in transitions if len(t)==2}
        while True:
            newly_connected_to_initial = {t[1] for t in transitions 
                                          if len(t)==4 and
                                          t[0] in connected_to_initial and
                                          t[1] not in connected_to_initial}
            if newly_connected_to_initial==set():
                break
            else:
                connected_to_initial.update(newly_connected_to_initial)
        while True:
            newly_connected_to_final = {t[0] for t in transitions 
                                          if len(t)==4 and
                                          t[1] in connected_to_final and
                                          t[0] not in connected_to_final}
            if newly_connected_to_final==set():
                break
            else:
                connected_to_final.update(newly_connected_to_final)
        connected_to_initial_and_to_final = connected_to_initial & connected_to_final
        pruned_transitions = set()
        for t in transitions:
            if (len(t)==4 
                and t[0] in connected_to_initial_and_to_final
                and t[1] in connected_to_initial_and_to_final):
                    pruned_transitions.add(t)
            if len(t)==2 and t[0] in connected_to_initial_and_to_final:
                pruned_transitions.add(t)
        if not pruned_transitions:
            # All transitions have been pruned !
            raise EmptyFsaError(self)
        return pruned_transitions       
    
    def prune_small_weights_transitions(self):
        MINIMAL_WEIGHT = 0.00000000000001 # Only used when not in LSR mode
        state_weights = self.accumulate_weights()
        weight_pruned_transitions = ( {(t[0],t[1],t[2],t[3]) for t in self.transitions if len(t)==4 and (LSR or (state_weights[t[0]] > MINIMAL_WEIGHT))} |
                                      {t for t in self.transitions if len(t)==2} )
        return weight_pruned_transitions
        
        
    
    def accumulate_weights(self):
        '''ToDo: this tolerance business is not well done here'''
        MAX_ITERATIONS = 10000
        if not LSR:
            EPSILON_TOLERANCE = 0.000000000001
        if LSR:
            EPSILON_TOLERANCE = 0.00000001 # This is probably not very good: not very demanding for large probs, very demanding for tiny probs
        accumulated_weights = {state: None for state in self.states}
        
        previous_accumulated_weights = {state: self.final_states_dict[state] if self.is_final_state(state) else gconvert(0.0) for state in self.states}
        for iteration in range(MAX_ITERATIONS):
            max_difference = 0.0
            for state in self.states:
                total_weight = gconvert(0.0)
                for e in self.transitions_dict[state]:
                    if len(e) == 1: # final state
                        # total_weight += e[0]
                        total_weight = gsum(total_weight,e[0])
                    if len(e) == 3:
                        # total_weight += previous_accumulated_weights[e[0]] * e[2]
                        total_weight = gsum(total_weight, gprod(previous_accumulated_weights[e[0]], e[2]))
                accumulated_weights[state] = total_weight
                # TO BE CHECKED: can we keep the formulation below if working in the LSR domain ?
                # print(f'iteration {iteration}, state {state}, max_dif {max_difference}, previous_acc {previous_accumulated_weights}, acc {accumulated_weights}')
                previous_accumulated_weights_for_state = previous_accumulated_weights[state]
                accumulated_weights_for_state = accumulated_weights[state]
                # This condition is to avoid a warning when subtracting -inf from -inf
                if accumulated_weights_for_state == float('-inf'):
                    acc_diff_for_state = 0.0
                else:
                    acc_diff_for_state = accumulated_weights_for_state - previous_accumulated_weights_for_state
                accumulated_weights_difference_for_state = abs(acc_diff_for_state)
                max_difference = max(max_difference, accumulated_weights_difference_for_state)
                # print(f'max_dif_after {max_difference}')
            if max_difference < EPSILON_TOLERANCE:
                break
            else:
                # Forgetting to copy leads to a difficult to catch error !!!!
                previous_accumulated_weights = copy.copy(accumulated_weights)
        return accumulated_weights

    def make_pfsa(self):
        '''Note: the input must be pruned'''
        accumulated_weights=self.accumulate_weights()
        
        partition = accumulated_weights[self.initial_state]
        
        new_transitions=set()
        for state, nexts in self.transitions_dict.items():
            total_nexts = gconvert(0.0)
            for next in nexts:
                if len(next)==3:
                    # contribution = next[2]*accumulated_weights[next[0]]
                    contribution = gprod(next[2], accumulated_weights[next[0]])
                if len(next)==1:
                    contribution = next[0]
                # total_nexts+=contribution
                total_nexts = gsum(total_nexts, contribution)
            # This does not seem to be correct for LSR:
            assert not math.isclose(convert_to_standard_real(total_nexts),0.0), "We may be trying to divide by a number close to 0 in the next lines"
            for next in nexts:
                if len(next)==3:
                    # new_transition=(state,next[0],next[1],(next[2]*accumulated_weights[next[0]])/total_nexts)
                    new_transition=(state,next[0],next[1], gprod(next[2], gdiv(accumulated_weights[next[0]], total_nexts)))
                if len(next)==1:
                    # new_transition=(state,next[0]/total_nexts)
                    new_transition=(state, gdiv(next[0], total_nexts))
                new_transitions.add(new_transition)
        return PFSA(initial_state=self.initial_state,transitions=new_transitions), partition

class EmptyFsaError(ValueError):
    pass


# # PFSA

# **Entropy of a PFSA**
# 
# We can compute the entropy of a (deterministic) PFSA through the fixpoint equation. Let $q$ be a state of the automaton, $(q, q', l, w)$ the transitions have source $q$, target $q'$, label $l$, and probability $w$ that have their source at $q$. It is easy to show that the entropy $H(q)$, namely the entropy of all the strings that start at $q$, satisfies the following equality: 
# 
# $$H(q) = \sum_{(q, q', l, w)} [- w \log w + w H(q')] .$$
# 
# We exploit this property in the entropy method of the following algorithm.

# In[12]:


class PFSA(WFSA):
    def __init__(self, initial_state='initial_state', transitions=set()):
        super().__init__(initial_state=initial_state, transitions=transitions)
        self.check_pfsa()

    def check_pfsa(self):
        for state,value in self.transitions_dict.items():
            sum = gconvert(0)
            for e in value:
                if len(e) == 3:
                    # sum += e[2]
                    sum = gsum(sum,e[2])
                if len(e)==1:
                    # sum += e[0]
                    sum = gsum(sum,e[0])
            try:
                assert math.isclose(convert_to_standard_real(sum),1.0), "Error: The state {0} is not normalized --- not a pfsa !".format(state)
            except AssertionError as err:
                print(err)
                # sys.exit()
        
    def sample(self):
        generated_labels = []
        source = self.initial_state
        while True:
            nexts = list(self.transitions_dict[source])
            probs = [convert_to_standard_real(next[2]) if len(next)==3 else convert_to_standard_real(next[0]) for next in nexts]
            i = np.random.choice(len(probs),p=probs)
            next = nexts[i]
            if len(next)==1: # we are exiting a final state
                break
            else:
                label = next[1]
                source = next[0]
                generated_labels.append(label)
        return generated_labels

    def entropy(self, base=2):
        '''
        Works only if the automaton is deterministic
        '''
        MAX_ITERATIONS = 10000 
        TOLERANCE = 0.00000000001
        # TOLERANCE = 0.0
        entropy_from_initial = 0.0
        # entropy_pair is (current_entropy, previous_entropy) --- two versions for current and previous iteration
        entropies = {state: (0.0,0.0) for state in self.states}
        for iteration in range(MAX_ITERATIONS):
            largest_difference = 0.0 # largest difference of entropies across states between current and previous iteration
            for state in self.states:
                nexts = self.transitions_dict[state]
                probs = [convert_to_standard_real(next[2]) if len(next)==3 else convert_to_standard_real(next[0]) for next in nexts]
                immediate_entropy = -sum([p * math.log(p,base) for p in probs if p != 0.0])
                next_entropies = [entropies[next[0]][0] if len(next)==3 else 0.0 for next in nexts]
                further_entropy = 0.0
                for i in range(len(probs)):
                    p = probs[i]
                    ent = next_entropies[i]
                    further_entropy += p * ent
                entropies[state] = (immediate_entropy+further_entropy, entropies[state][0])
                after, before = entropies[state]
                largest_difference = max(largest_difference, abs(after-before))
                # print(f'iteration: {iteration}, state: {state}, after: {after}, before: {before}, largest_difference: {largest_difference}')
            # NOT SURE ABOUT THIS !!!!! TO CHECK
            if largest_difference < TOLERANCE:
                break
        print(f'entropy iterations: {iteration}')
        entropy_from_initial = entropies[self.initial_state][0]
        return entropy_from_initial
    
    def mean_length(self): # 2.04.2019 HHHHHHHHHH
        MAX_ITERATIONS = 10000 
        TOLERANCE = 0.00000001
        mean_lengths = {state: (0.0,0.0) for state in self.states}
        for iteration in range(MAX_ITERATIONS):
            largest_difference = 0.0 # largest difference of mean_lengths across states between current and previous iteration
            for state in self.states:
                nexts = self.transitions_dict[state]
                probs = [convert_to_standard_real(next[2]) if len(next)==3 else convert_to_standard_real(next[0]) for next in nexts]
                next_mean_lengths = [1+mean_lengths[next[0]][0] if len(next)==3 else 0.0 for next in nexts]
                current_mean_length = 0.0
                for i in range(len(probs)):
                    p = probs[i]
                    current_mean_length += p * next_mean_lengths[i]
                mean_lengths[state] = (current_mean_length, mean_lengths[state][0])
                after, before = mean_lengths[state]
                largest_difference = max(largest_difference, abs(after-before))
            if largest_difference < TOLERANCE:
                break
        print(f'mean_length iterations: {iteration}')
        mean_length_from_initial = mean_lengths[self.initial_state][0]
        return mean_length_from_initial
         

# # Intersection

# In[23]:


def wfsa_intersection(A,B):
    initial_state=(A.initial_state,B.initial_state)
    transitions_len4 = {((ta[0],tb[0]),(ta[1],tb[1]),ta[2],gprod(ta[3],tb[3])) for ta in A.transitions for tb in B.transitions if len(ta)==4 and len(tb)==4 and ta[2]==tb[2]}
    transitions_len2 = {((ta[0],tb[0]), gprod(ta[1],tb[1])) for ta in A.transitions for tb in B.transitions if len(ta)==2 and len(tb)==2}
    transitions= transitions_len4 | transitions_len2
    return WFSA(initial_state=initial_state,transitions=transitions)


def pattern_to_wfsa(pattern,vocab):
    '''
    pattern: a list of symbols from the set vocab

    Output: a WFSA that assigns a weight 1 to strings (as lists) that contain pattern as a sublist
    and that end with '$'
    
    Note: $ is not explicitly given in the vocab.
    '''
    def pattern2triples(pattern,vocab):
        # init=[], final=pattern
        prefixes = pattern2prefixes(pattern) # in decreasing order
        states = prefixes
        triples = []
        for state in states:
            if state == pattern: # we are already at the final state
                for symbol in vocab:
                    triples.append((state,symbol,state))
            else:
                for symbol in vocab:
                    extended_state = state + [symbol]
                    for prefix in prefixes:
                        if is_suffix(prefix,extended_state):
                            triples.append((state,symbol,prefix))
                            break    
        return triples
    
    def pattern2prefixes(pattern):
        prefixes = []
        for i in reversed(range(len(pattern)+1)):
            prefix = pattern[0:i]
            prefixes.append(prefix)
        return prefixes
    
    def is_suffix(list1, list2):
        list2_reversed=list(reversed(list2))
        list1_reversed=list(reversed(list1))
        list2_reversed_prefixes = pattern2prefixes(list2_reversed)
        if list1_reversed in list2_reversed_prefixes:
            return True
        else:
            return False
        
    triples = pattern2triples(pattern,vocab)
    triples_with_state_numbers = [(len(triple[0]),triple[1],len(triple[2])) for triple in triples]
    lp = len(pattern) 
    initial = 0
    prefinal = lp
    final = lp+1
    ######### MD : Tue 30 April 2019: the next line looks like a mistake: should use gconvert ?
    #transitions = {(t[0],t[2],t[1],1.0) for t in triples_with_state_numbers}
    transitions = {(t[0],t[2],t[1],gconvert(1.0)) for t in triples_with_state_numbers}
    transitions.add((prefinal,final,'$',gconvert(1.0)))
    transitions.add((final,gconvert(1.0)))
    
    return WFSA(initial_state=initial,transitions=transitions)

# In[36]:


def anti_pattern_to_wfsa(pattern,vocab_minus_dollar):
    all_vocab = vocab_minus_dollar | {'$'}
    wfsa1 = pattern_to_wfsa(pattern,vocab_minus_dollar)
    # wfsa1.display(verbosity=1)
    td1 = wfsa1.transitions_dict
    # print(td1)
    new_final_state = "<cfs>" # completion final state
    new_final_states = {new_final_state}
    new_non_final_states = set()
    new_transitions = {(new_final_state,gconvert(1.0))}
    for state in td1.keys():
        nexts = td1[state]
        # print(nexts)
        labels = {suivant[1] for suivant in nexts if len(suivant) == 3}
        other_labels = all_vocab - labels  # for completion of dfa
        if {next for next in nexts if len(next) == 1}:
            new_nexts = nexts - {next for next in nexts if len(next) == 1} # a final state become non final
            new_non_final_states = new_non_final_states | {state}
        else:
            new_nexts = nexts | {(gconvert(1.0),)} # a non final becomes final
            new_final_states = new_final_states | {state}
        new_nexts = new_nexts | {(new_final_state,other_label,gconvert(1.0)) for other_label in other_labels}
        new_transitions = new_transitions | {(state,) + next for next in new_nexts}

    
    
    # print('new_final_states: ', new_final_states)
    # print('new_non_final_states: ', new_non_final_states)
    # pprint(new_transitions)
    
    # remove dollar transitions to non-final state
    new_transitions = new_transitions - {t for t in new_transitions if len(t) == 4 and t[2] == '$' and t[1] in new_non_final_states}
    
    
    # print('bibi: ', new_transitions)


    return WFSA(initial_state=wfsa1.initial_state,transitions=new_transitions)


# In[37]:


# # White Noise

# In[41]:


def white_noise_pfsa(n,prob0=0.5):
    "n is the fixed length of the sequence"
    transitions = (
        {(i,i+1,'0',gconvert(prob0)) for i in range(n)} | 
        {(i,i+1,'1',gconvert(1-prob0)) for i in range(n)} |  
        {(n,n+1,'$',gconvert(1.0))} |
        {(n+1,gconvert(1.0))}
    )
    return PFSA(0,transitions)


# ### Geometric white noise (added: 2.04.2019)

# In[42]:


def geometric_white_noise_pfsa(prob_0=0.4,prob_1=0.4,prob_dollar=0.2):
    transitions = ((0, 0, '0', gconvert(prob_0)),
                   (0, 0, '1', gconvert(prob_1)),
                   (0, 1, '$', gconvert(prob_dollar)),
                   (1, gconvert(1.0)))
    return PFSA(0,transitions)



# In[67]:


def make_wfsa_wn_pattern(length,pattern_as_string):
    wn_pfsa = white_noise_pfsa(length)
    pattern_automaton = pattern_to_wfsa(list(pattern_as_string),set('01'))
    wfsa_inter = wfsa_intersection(wn_pfsa,pattern_automaton)
    return wfsa_inter

def make_pfsa_wn_pattern(length,pattern_as_string):
    wfsa_inter = make_wfsa_wn_pattern(length,pattern_as_string)
    pfsa_inter, _ = wfsa_inter.make_pfsa()
    return pfsa_inter


def line_from_pfsa_sample(pfsa):
    sample = pfsa.sample()
    line = '# ' + ' '.join(['_' if x == '0' else '+' if x == '1' else x for x in sample]) + '\n'
    return line
    
# # Generator for data: second version; with separators

# In[84]:

def make_data_from_motif(args):
    os.makedirs(args.data_target,exist_ok=True)
    if args.bare_seq_length:
        base_pfsa = white_noise_pfsa(args.bare_seq_length, prob0=args.base_prob0)
    else:
        base_pfsa = geometric_white_noise_pfsa(prob_0=args.base_prob0, prob_1=args.base_prob1, prob_dollar=args.base_prob_dollar)
    if args.pfsa:
        pfsa=args.pfsa
    elif not args.motif:
        pfsa=base_pfsa
    else:
        wfsa_motif = pattern_to_wfsa(list(args.motif),set('01'))
        # wfsa_motif.display()
        wfsa_inter = wfsa_intersection(base_pfsa,wfsa_motif)
        # wfsa_inter.display()
        pfsa, _ = wfsa_inter.make_pfsa()
        # pfsa.display()
    pfsa_entropy = pfsa.entropy(base=math.e)
    # 3.04.2019: commented out next two lines. Do later.
    # entropy_per_symbol = pfsa_entropy / args.bare_seq_length
    # perplexity_per_symbol = math.exp(entropy_per_symbol)
    print(f'nats per sequence: {pfsa_entropy}') # , nats per symbol: {entropy_per_symbol}')
    print(f'bits per sequence: {pfsa_entropy * math.log2(math.e)}') # , bits per symbol: {entropy_per_symbol * math.log2(math.e)}')
    pfsa_mean_length = pfsa.mean_length()
    print(f'mean sequence length (including $): {pfsa_mean_length}')
    
    if args.separators_included:
        # entropy_per_actual_symbol = pfsa_entropy / (args.bare_seq_length + 3) # We add 3 symbols (# $ <eos>) and I think this is correct
        # perplexity_per_actual_symbol = math.exp(entropy_per_actual_symbol)
        # print(f'entropy_per_actual_symbol (bits): {entropy_per_actual_symbol}\nperplexity_per_actual_symbol: {perplexity_per_actual_symbol}')
        # print(f'Note: "actual" symbols also include the separator symbols')
        pfsa_mean_length = pfsa.mean_length()
        print(f'mean sequence length (including $): {pfsa_mean_length}')


    
    def line_from_pfsa_sample():
        sample = pfsa.sample()
        if args.separators_included:
            line = '# ' + ' '.join(['_' if x == '0' else '+' if x == '1' else x for x in sample]) + '\n'
        else:
            line = ''.join(['0' if x == '0' else '1' if x == '1' else '' if x == '$' else x for x in sample]) + '\n'
        return line
    
    def make_data_file(filename,number_lines):
        with open(filename, "w") as f:
            for _ in range(number_lines):
                f.write(line_from_pfsa_sample())
                

    data_target = args.data_target
    train, valid, test = (data_target+'/train.txt', data_target+'/valid.txt', data_target+'/test.txt')
    
    make_data_file(train, args.train_number)
    make_data_file(valid, args.valid_number)
    make_data_file(test, args.test_number)
    
    return None
    
# In[101]:

def motif2data(pfsa_name='some_pfsa_name',length=10,motif='0',data_dir='./some_psa_name',train_number=10000,valid_number=1000,test_number=1000):
    namespace = Namespace(
                 pfsa_name=pfsa_name,
                 pfsa=None,
                 base_prob0=0.5,
                 bare_seq_length=length,
                 motif=motif,
                 separators_included=False,
                 data_target=data_dir,
                 train_number=train_number,
                 valid_number=valid_number,
                 test_number=test_number
            )
    make_data_from_motif(args=namespace)
    return None

def wfsa_selection(A,B,w0=0.5,w1=0.5,initial_state=0):
    transitions_len4_from_A = {(('A',ta[0]),('A',ta[1]),ta[2],ta[3]) for ta in A.transitions if len(ta)==4} 
    transitions_len4_from_B = {(('B',tb[0]),('B',tb[1]),tb[2],tb[3]) for tb in B.transitions if len(tb)==4}
    transitions_len4 = {(initial_state,('A',A.initial_state),'0',gconvert(w0))}|{(initial_state,('B',B.initial_state),'1',gconvert(w1))}|transitions_len4_from_A|transitions_len4_from_B
                        
    transitions_len2_from_A = {(('A',ta[0]), ta[1]) for ta in A.transitions if len(ta)==2}
    transitions_len2_from_B = {(('B',tb[0]), tb[1]) for tb in B.transitions if len(tb)==2}
    transitions_len2 = transitions_len2_from_A | transitions_len2_from_B
    transitions= transitions_len4 | transitions_len2
    # print(transitions)
    return WFSA(initial_state=initial_state,transitions=transitions)


# In[111]:

def make_data_from_pfsa(pfsa, train_number=1000, valid_number=100, test_number=100, data_target='./data', separators_included=False,
                       second_selector_bit_remove = False):
    
    def line_from_pfsa_sample():
        sample = pfsa.sample()
        if separators_included:
            line = '# ' + ' '.join(['_' if x == '0' else '+' if x == '1' else x for x in sample]) + '\n'
        else:
            line = ''.join(['0' if x == '0' else '1' if x == '1' else '' if x == '$' else x for x in sample]) + '\n'
        return line
    
    def make_data_file(filename,number_lines):
        with open(filename, "w") as f:
            for _ in range(number_lines):
                line = line_from_pfsa_sample()
                if second_selector_bit_remove:
                    line=line[1:]
                f.write(line)
                
    os.makedirs(data_target,exist_ok=True)

    train, valid, test = (data_target+'/train.txt', data_target+'/valid.txt', data_target+'/test.txt')
    
    make_data_file(train, train_number)
    make_data_file(valid, valid_number)
    make_data_file(test, test_number)
    
    return None


# In[112]:


def make_pfsa(args):
    '''
    This function is consistent with the following command-line arguments:
    
    parser.add_argument("-length", type=int, help="Fixed length of binary string; absence means variable length")
    parser.add_argument("-prob_0", type=float, help="Prob of '0' in var length string")
    # parser.add_argument("-prob_1", type=float, help="Prob of '1' in var length string")
    parser.add_argument("-prob_dollar", type=float, help="Prob of '$' in var length string")            
    parser.add_argument("-motif", type=str, help="motif string; if absent, no motif is used")

    parser.add_argument("-second_select_prob", type=float, help="probability of selecting second automaton")
    parser.add_argument("-second_length", type=int, help="Fixed length of binary string; absence means variable length")
    parser.add_argument("-second_prob_0", type=float, help="Prob of '0' in var length string")
    # parser.add_argument("-second_prob_1", type=float, help="Prob of '1' in var length string")
    parser.add_argument("-second_prob_dollar", type=float, help="Prob of '$' in var length string")            
    parser.add_argument("-second_motif", type=str, help="motif string; if absent, no motif is used")
    '''
    if args.length:
        wn1 = white_noise_pfsa(args.length, prob0=args.prob_0)
    else:
        wn1 = geometric_white_noise_pfsa(prob_0=args.prob_0, prob_1=1.0-(args.prob_0 + args.prob_dollar), prob_dollar=args.prob_dollar)
    if args.motif:
        motif1 = pattern_to_wfsa(list(args.motif),set('01'))
        wfsa1 = wfsa_intersection(wn1,motif1)
        pfsa1,partition1 = wfsa1.make_pfsa()
    elif args.anti_motif:
        anti_motif1 = anti_pattern_to_wfsa(list(args.anti_motif),set('01'))
        wfsa1 = wfsa_intersection(wn1,anti_motif1)
        pfsa1, partition1 = wfsa1.make_pfsa()
    else:
        pfsa1 = wn1
        partition1 = gconvert(1.0)
    print(f'Partition function for wfsa1: {partition1}')
    if not args.second_select_prob:
        pfsa = pfsa1
    else:
        if args.second_length:
            wn2 = white_noise_pfsa(args.second_length, prob0=args.second_prob_0)
        else:
            wn2 = geometric_white_noise_pfsa(prob_0=args.second_prob_0, prob_1=1.0-(args.second_prob_0 + args.second_prob_dollar), prob_dollar=args.second_prob_dollar)
        if args.second_motif:
            motif2 = pattern_to_wfsa(list(args.second_motif),set('01'))
            wfsa2 = wfsa_intersection(wn2,motif2)
            pfsa2, partition2 = wfsa2.make_pfsa()
        elif args.second_anti_motif:
            anti_motif2 = anti_pattern_to_wfsa(list(args.second_anti_motif),set('01'))
            wfsa2 = wfsa_intersection(wn2,anti_motif2)
            pfsa2, partition2 = wfsa2.make_pfsa()
        else:
            pfsa2 = wn2
            partition2 = gconvert(1.0)
        print(f'Partition function for wfsa2: {partition2}')
        wfsa = wfsa_selection(pfsa1, pfsa2, w0 = 1.0-args.second_select_prob, w1 = args.second_select_prob)
        pfsa, _ = wfsa.make_pfsa()

        if args.length == args.second_length and args.prob_0 == args.second_prob_0:
            partition = (1 - args.second_select_prob)*partition1 + args.second_select_prob*partition2
            print(f'Partition function for wfsa1 and wfsa2: {partition}')
    
        
        
    mean_length = pfsa.mean_length()
    entropy = pfsa.entropy(base=math.e)
    print(f'mean sequence length (including $ plus second_selector_bit in case of second automaton): {mean_length}')
    print(f'entropy in nats per sequence: {entropy}; in bits per sequence: {entropy / math.log(2)}')

    print('number of strings: {:.4f}, ratio {:.4f}'.format(np.exp(entropy), 2**args.length/np.exp(entropy)))
        
    return pfsa


# In[114]:


def make_data_from_specif(args):
    pfsa = make_pfsa(args)
    make_data_from_pfsa(pfsa,
                        train_number=args.train_number, 
                        valid_number=args.valid_number, 
                        test_number=args.test_number, 
                        data_target=args.data_target, 
                        separators_included=False,
                        second_selector_bit_remove=args.second_selector_bit_remove)





if __name__ == '__main__':

    example_text = """Usage examples:

    python <name_of_this_script> -length 10 -prob_0 0.99  -train 5000 -valid 500 -test 500 -data_target ./data/example2

    python <name_of_this_script> -prob_0 0.495  -prob_dollar 0.01 -motif 101000110  -second_select_prob 0.3 -second_prob_0 0.1 -second_prob_dollar 0.1  -train 5000 -valid 500 -test 500 -data_target ./data/example1    

    # anti_motif means that the second automaton does not produce strings containing the motif
    python <name_of_this_script>is selected with probability second_select_prob, 
    a 'selector_bit' prefix (for first automaton) of '0' with 1 minus this probability,
    and where each of the two automata has its own description (fixed or variable length, motif or not).

    Note that prob_1 is computed as 1-prob_0 in the case of a fixed
    length sequence, and as 1-(prob_0 + prob_dollar) in the case of a
    variable length seqquence.

    Generally speaking, presence of different arguments have to
    satisfy certain consistency rules, which can be summarized by the
    following regular expression (only the arguments, not their values, are indicated):

    (-length -prob_0 | -prob_0 -prob_dollar) 
     (-motif | -anti_motif)? 
      ( -second_select_prob (-second_length -second_prob_0 | -second_prob_0 -second_prob_dollar) (-second_motif | -second_anti_motif)? -second_selector_bit_remove? )?

  """
    
    parser = argparse.ArgumentParser(description= 'Process motif and create data sets',
                                     epilog = example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-log_domain", action='store_true', help="If present, log domain is used; else standard real probs")
    parser.add_argument("-length", type=int, help="Fixed length of binary string; absence means variable length")
    parser.add_argument("-prob_0", type=float, help="Prob of '0' in fixed or variable length string")
    parser.add_argument("-prob_dollar", type=float, help="Prob of '$' in variable length string")            
    parser.add_argument("-motif", type=str, help="motif string; if absent, no motif is used")
    parser.add_argument("-anti_motif", type=str, help="anti_motif string; if absent, no anti_motif is used")

    parser.add_argument("-second_select_prob", type=float, help="probability of selecting second automaton")
    parser.add_argument("-second_length", type=int, help="Fixed length of binary string; absence means variable length")
    parser.add_argument("-second_prob_0", type=float, help="Prob of '0' in fixed or variable length string")
    parser.add_argument("-second_prob_dollar", type=float, help="Prob of '$' in variable length string")            
    parser.add_argument("-second_motif", type=str, help="motif string; if absent, no motif is used")
    parser.add_argument("-second_anti_motif", type=str, help="anti_motif string; if absent, no anti_motif is used")
    parser.add_argument("-second_selector_bit_remove", action='store_true', help="if present, the first 'selector' bit is removed")

    parser.add_argument("-data_target", type=str, default='./data/pfsa_', help="data directory prefix (default ./data/pfsa_  )")
    parser.add_argument('-train', '--train_number', default=10000, type=int, help="size of training set (default 10000)")

    parser.add_argument('-valid', '--valid_number', default=1000, type=int, help="size of validation set (default 1000)")
    parser.add_argument('-test', '--test_number', default=1000, type=int, help="size of test set (default 1000)")
    args = parser.parse_args()

    print(args)
    
    if args.log_domain:
        LSR = True
        print("Computations done in the log semiring (Probs represented as logs)")
    else:
        LSR = False
        print("Computations done in the standard semiring (Probs represented as reals)")
    make_data_from_specif(args)

    
# python wfsa_n_z.py -prob_0 0.5 -length 30 -motif 1011100111001  -data_target ./data/pfsa_30_1011100111001 -valid 2000 -test 5000 -train 20000

# python wfsa_m.py -prob_0 0.5 -length 30 -motif 10001011111000 -second_select_prob 0.1 -second_length 30 -second_prob_0 0.5 -second_anti_motif 10001011111000 -second_selector_bit_remove -data_target ./data/pfsa_30_10001011111000.10001011111000 -valid 2000 -test 5000 -train 20000