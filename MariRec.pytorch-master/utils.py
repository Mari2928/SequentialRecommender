import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, args):
      
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        # NextItem or NextAllItem
        if args.model == 1 or args.model == 2:          
          seq = np.zeros([maxlen], dtype=np.int32)
          pos = np.zeros([maxlen], dtype=np.int32)
          neg = np.zeros([maxlen], dtype=np.int32)
          nxt = user_train[user][-1]
          idx = maxlen - 1

          ts = set(user_train[user])
          for i in reversed(user_train[user][:-1]):
              seq[idx] = i
              pos[idx] = nxt
              if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
              nxt = i
              idx -= 1
              if idx == -1: break

        # FutureAllItem or FutureDenseAll
        else:
          N = args.window_size # window size
          S = args.rand_neg  # number of random negatives
          seq = np.zeros([maxlen], dtype=np.int32) 
          pos = np.zeros([maxlen, N], dtype=np.int32)
          neg = np.zeros([maxlen, S], dtype=np.int32)
          nxts= [user_train[user][-1] for _ in range(N)]
          idx = maxlen - 1

          ts = set(user_train[user])
          for i in reversed(user_train[user][:-1]):
              seq[idx] = i 
              pos[idx] = nxts
              if nxts[0] != 0: 
                neg[idx] = [random_neq(1, itemnum + 1, ts) for _ in range(S)]
              
              for j in range(len(nxts)-1, 0, -1):
                nxts[j] = nxts[j-1]
              nxts[0] = i      
              idx -= 1
              if idx == -1: break
        
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):       
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, args, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,                                           
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      args
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# Leave N Out: no fix of max length; for BOW output
def data_partition_all(fname, args):

    N = args.window_size
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < N+1:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-N-1]
            user_valid[user] = []
            user_valid[user].append(User[user][-N-1]) 
            # for i in range(-2*N, -N):
            #   user_valid[user].append(User[user][i])            
            user_test[user] = []
            for i in range(-N,0):
              user_test[user].append(User[user][i])
    return [user_train, user_valid, user_test, usernum, itemnum]

# Leave N Out With A Threshold
def data_partition_all2(fname, args):

    N = args.window_size    
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < args.max_window+1:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-args.max_window-1]
            user_valid[user] = []              
            user_valid[user].append(User[user][-args.max_window-1])       
            user_test[user] = [] 
            for i in range(-args.max_window,0):
              user_test[user].append(User[user][i])
    return [user_train, user_valid, user_test, usernum, itemnum]

# train/val/test data generation
def data_partition(fname, args):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        # original
        seq = np.zeros([args.maxlen], dtype=np.int32)        
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        # seq = np.zeros([args.maxlen, 2], dtype=np.int32)        
        # idx = args.maxlen - 1
        # for i in reversed(train[u]):
        #     seq[idx] = [i, i]
        #     idx -= 1
        #     if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# evaluate on test set for all actions prediction: sasrec split
def evaluate_all(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    N = args.window_size
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    NDCGs = np.array([0.0] * N)
    HTs = np.array([0.0] * N)

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        # take N test items
        test[u] = test[u][:N]  

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        train[u] += valid[u]
        rated = set(train[u])
        rated.add(0)

        # original
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break        

        item_idx = []
        for item in test[u]:
          item_idx.append([item])
        for idx in item_idx:
          for _ in range(100):
              t = np.random.randint(1, itemnum + 1)
              while t in rated: t = np.random.randint(1, itemnum + 1)
              idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])

        ranks = []
        for prediction in predictions:
          ranks.append(prediction.argsort().argsort()[0].item())

        valid_user += 1
        NDCG_user = 0.0
        HT_user = 0.0      
        
        for i in range(0,len(ranks)):
          if ranks[i] < 10:
            NDCGs[i] += 1 / np.log2(ranks[i] + 2)
            HTs[i] += 1

        for rank in ranks:
          if rank < 10:
              NDCG_user += 1 / np.log2(rank + 2)
              HT_user += 1
        # if HT_user != 0:
        NDCG += (NDCG_user / N)
        HT += (HT_user / N)

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, np.round(NDCGs / valid_user, 4), np.round(HTs / valid_user, 4)

# dense all sasrec split
def evaluate_denseall(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    N = args.window_size
    seq_num = 2
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    NDCGs = np.array([0.0] * N)
    HTs = np.array([0.0] * N)

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        # take N test items
        test[u] = test[u][:N]  

        if len(train[u]) < 1 or len(test[u]) < 1: continue  
        
        # create a sequence with whole positive items
        train[u] += valid[u]
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        rated = set(train[u])
        rated.add(0)   
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break                              
        
        # change train length decrementaly
        #tr_lengths = random.sample(range(len(train[u]) - N, len(train[u]) + 1), 5)
        l=len(train[u])
        tr_lengths = [l-i for i in range(1, N)]
        # stack multiple train sequences for a user
        for tr_len in tr_lengths:          
          seq_ = np.zeros([args.maxlen], dtype=np.int32)
          idx = args.maxlen - 1  
          train_u = train[u][:tr_len]
          for j in reversed(train_u):
              seq_[idx] = j
              idx -= 1
              if idx == -1: break
          seq = np.vstack((seq, seq_))

        # predict each test item with multiple sequences
        #ranks = [0.0 for _ in range(N)]
        # for i in range(0, len(test[u])):        
        #   item_idx = [test[u][i]]
        #   for _ in range(100):
        #       t = np.random.randint(1, itemnum + 1)
        #       while t in rated: t = np.random.randint(1, itemnum + 1)
        #       item_idx.append(t)
          # prediction = -model.predict(*[np.array(l) for l in [[u], [seq[idx]], item_idx]])
          # ranks[i] = prediction[0].argsort().argsort()[0].item() 

        item_idx = []
        for item in test[u]:
          item_idx.append([item])
        for idx in item_idx:
          for _ in range(100):
              t = np.random.randint(1, itemnum + 1)
              while t in rated: t = np.random.randint(1, itemnum + 1)
              idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], seq, item_idx]])

        ranks = []
        for prediction in predictions:
          ranks.append(prediction.argsort().argsort()[0].item())

        valid_user += 1
        NDCG_user = 0.0
        HT_user = 0.0    

        for i in range(0,len(ranks)):
          if ranks[i] < 10:
            NDCGs[i] += 1 / np.log2(ranks[i] + 2)
            HTs[i] += 1

        for rank in ranks:
          if rank < 10:
              NDCG_user += 1 / np.log2(rank + 2)
              HT_user += 1

        NDCG += (NDCG_user / N)
        HT += (HT_user / N)

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, np.round(NDCGs / valid_user, 4), np.round(HTs / valid_user, 4)

def evaluate_next(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    N = args.window_size
    seq_num = 2
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    NDCGs = np.array([0.0] * N)
    HTs = np.array([0.0] * N)

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        # take N test items
        test[u] = test[u][:N]  

        if len(train[u]) < 1 or len(test[u]) < 1: continue  
        
        # create a sequence with whole positive items
        train[u] += valid[u]                        

        # predict each test item with a seq + predicted item
        ranks = []
        for i in range(0, len(test[u])):   
          seq = np.zeros([args.maxlen], dtype=np.int32)
          idx = args.maxlen - 1
          rated = set(train[u])
          rated.add(0)  
          for j in reversed(train[u]):
            seq[idx] = j
            idx -= 1
            if idx == -1: break     

          item_idx = [test[u][i]]
          for _ in range(100):
              t = np.random.randint(1, itemnum + 1)
              while t in rated: t = np.random.randint(1, itemnum + 1)
              item_idx.append(t)

          prediction = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
          pred = prediction[0].argsort().argsort()
          ranks.append(pred[0].item())  
          # add predicted item to seq
          train[u].append(item_idx[(pred == 0).nonzero(as_tuple=True)[0]])
          
        valid_user += 1
        NDCG_user = 0.0
        HT_user = 0.0    

        for i in range(0,len(ranks)):
          if ranks[i] < 10:
            NDCGs[i] += 1 / np.log2(ranks[i] + 2)
            HTs[i] += 1

        for rank in ranks:
          if rank < 10:
              NDCG_user += 1 / np.log2(rank + 2)
              HT_user += 1

        NDCG += (NDCG_user / N)
        HT += (HT_user / N)

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, np.round(NDCGs / valid_user, 4), np.round(HTs / valid_user, 4)
