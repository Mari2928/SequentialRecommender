import os
import time
import torch
import argparse
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from model import SASRec
from utils import *

def str2bool(s):
  if s not in {'false', 'true'}:
      raise ValueError('Not a valid boolean string')
  return s == 'true'

# logQ for random negatives
def rand_neg_logQ(pos, neg, itemnum):  
  logQ_n = {}
  logQ_p = {}
  logQ_p_list = [0.0 for i in range(itemnum+1)]  
  logQ_n_list = [0.0 for i in range(itemnum+1)]  

  # create a freq dict for positives
  # for i in range(pos.shape[0]): # 128    
  #   for j in range(pos.shape[1]):  # 200
  #     for k in range(pos.shape[2]):  # 8
  #       item_p = pos[i,j,k]
        
  #       if (item_p == 0):
  #         logQ_p[item_p] = 0
  #       elif (item_p in logQ_p):
  #         logQ_p[item_p] += 1
  #       else:
  #         logQ_p[item_p] = 1

  # create a freq dict for negatives
  for i in range(neg.shape[0]): # 128    
    for j in range(neg.shape[1]):  # 200
      for k in range(neg.shape[2]):  # 8
        item_n = neg[i,j,k]

        if (item_n == 0):
          logQ_n[item_n] = 0
        elif (item_n in logQ_n):
          logQ_n[item_n] += 1
        else:
          logQ_n[item_n] = 1
    
  # expectation of probabilities for positives
  # for idx in logQ_p:
  #   if idx == 0: continue
  #   prob = logQ_p.get(idx) / itemnum
  #   logQ_p_list[idx] = np.log(prob)


  # expectation of probabilities for negatives
  for idx in logQ_n:
    if idx == 0: continue
    prob = logQ_n.get(idx) / itemnum
    logQ_n_list[idx] = np.log(prob)

  return torch.FloatTensor(logQ_n_list)

# add N in-batch negatives
def mix_inbatch_neg(pos, neg, itemnum, n_negs):
  sampled_items=[]
  freq = {}
  batch = set()    
  p_user = [set()  for _ in range(pos.shape[1])]  
  logQ_list = [0.0 for i in range(itemnum+1)]  

  for i in range(pos.shape[0]): # batch 
    for j in range(pos.shape[1]):  # user
      for k in range(pos.shape[2]):  # item
        item = pos[i,j,k]
        p_user[j].add(item)
        batch.add(item)  
        
  batch = list(batch)
  neg_ = []
  for i in range(pos.shape[0]): # 128   
    itmnegs = []
    for j in range(pos.shape[1]): # 200
      n = []
      size = 0
      while size != n_negs:
        t = random.choices(batch, k=1)[0]
        if t in p_user[j]: continue
        n.append(t)
        size += 1  
        sampled_items.append(t)

        # create a freq dict 
        if (t == 0):
          freq[t] = 0
        elif (t in freq):
          freq[t] += 1
        else:
          freq[t] = 1
      itmnegs.append(np.append(neg[i][j], n))
    neg_.append(itmnegs)

  # expectation of probabilities for in-batch negatives
  for idx in freq:
    if idx == 0: continue
    prob = freq.get(idx) / itemnum
    logQ_list[idx] = np.log(prob)

  # candidate probabilities
  # for idx in freq:
  #   prob = (np.log(idx+2) - np.log(idx+1)) / np.log(len(s)+1)
  #   freq[idx] = -np.exp(25600 * np.log(1+(-prob))) - 1

  return np.array(neg_), torch.FloatTensor(logQ_list), sampled_items

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--window_size', default=2, type=int)
parser.add_argument('--num_items', default=3, type=int)
parser.add_argument('--max_window', default=28, type=int)
parser.add_argument('--batch_neg', default=0, type=int)
parser.add_argument('--rand_neg', default=8, type=int)
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--logq', default=0, type=int)
parser.add_argument('--input_window', default=3, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    if args.model == 2: # Leave N Out 
      dataset = data_partition_all(args.dataset, args)
    else: # Leave N Out With Threshold
      dataset = data_partition_all2(args.dataset, args)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # with open('user_train_ml', 'wb') as f3:
    #   pickle.dump(user_train, f3)

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(args, user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate_all(model, dataset, args)
        #print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        print('test:',  t_test)
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray  
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)    

            logQ_inbatchn_list=[]; logQ_randn_list=[]
            if args.logq == 1:
              logQ_randn_list = rand_neg_logQ(pos, neg, itemnum)

            if args.batch_neg != 0:
              neg, logQ_inbatchn_list, sampled_items = mix_inbatch_neg(pos, neg, itemnum, args.batch_neg)  

            if args.model != 4:
              pos_logits, neg_logits = model(u, seq, pos, neg, logQ_inbatchn_list, logQ_randn_list)
            else: # for denseall
              all_logits, all_labels = model(u, seq, pos, neg, [], []) 

            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()

            ### NEXT ITEM ##############
            if args.model == 1 or args.model == 2:            
              pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
              indices = np.where(pos != 0)   
              
              logits = torch.cat((pos_logits[indices], neg_logits[indices]), dim=0) # Softmax
              labels = torch.cat((pos_labels[indices], neg_labels[indices]), dim=0)
              loss = ce_criterion(logits, labels)
                
              # loss = bce_criterion(pos_logits[indices], pos_labels[indices])# BCE   
              # loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            ### FUTURE ALL ITEM ############   
            elif args.model == 3:     
              loss = 0
              for i in range(args.window_size):
                pos_labels = torch.ones(pos_logits[i].shape, device=args.device)          
                neg_labels = torch.zeros(pos_logits[i].shape, device=args.device)

                indices = np.where(pos[:,:,i] != 0) 
                logits = torch.cat((pos_logits[i][indices], neg_logits[0][indices]), dim=0) 
                labels = torch.cat((pos_labels[indices], neg_labels[indices]), dim=0)
                for j in range(1, len(neg_logits)):
                  logits = torch.cat((logits, neg_logits[j][indices] ), dim=0) 
                  labels = torch.cat((labels, neg_labels[indices]), dim=0)
                loss += ce_criterion(logits, labels)
              loss = loss.mean()  

            ### FUTURE ITEMS DENSE ALL ITEMS ############  
            elif args.model == 4: 
              loss = 0
              for logits, labels in zip(all_logits, all_labels):
                loss += ce_criterion(logits, labels)               
              loss = loss.mean()          
            #############################################         
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()        
            adam_optimizer.step()
            #print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            if args.model == 1: 
              t_test = evaluate_next(model, dataset, args)
            else:
              t_test = evaluate_all(model, dataset, args)
            #t_valid = evaluate_valid(model, dataset, args)

            #print('valid:',  t_valid)
            print('test:',  t_test)    

            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
            # with open('sampled_items', 'wb') as f2:
            #   pickle.dump(sampled_items, f2)
    
    f.close()
    sampler.close()
    print("Done")
