import os
import time
import torch
import pickle
import argparse

from model import TiSASRec
from tqdm import tqdm
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# add N in-batch negatives
def mix_inbatch_neg(pos, neg, itemnum, n_negs):
  freq = {}
  batch = set()    
  p_user = [set()  for _ in range(pos.shape[1])]  
  logQ_list = [0.0 for i in range(itemnum+1)]  

  for i in range(pos.shape[0]): # batch 
    for j in range(pos.shape[1]):  # query
      for k in range(pos.shape[2]):  # positives
        item = pos[i,j,k]
        p_user[j].add(item)
        batch.add(item)  
        
  batch = list(batch)
  neg_ = []
  for i in range(pos.shape[0]): # 128   
    itmnegs = []
    for j in range(pos.shape[1]): # 50
      n = []
      size = 0
      while size != n_negs:
        t = random.choices(batch, k=1)[0]
        if t in p_user[j]: continue
        n.append(t)
        size += 1  

        # create a freq dict 
        if (t == 0):
          freq[t] = 0
        elif (t in freq):
          freq[t] += 1
        else:
          freq[t] = 1
      itmnegs.append(np.append(neg[i][j], n))
    neg_.append(itmnegs)
  
  for idx in freq:
    if idx == 0: continue
    prob = freq.get(idx) / itemnum
    logQ_list[idx] = np.log(prob)
  
  return np.array(neg_), torch.FloatTensor(logQ_list) 

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
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--num_items', default=3, type=int)
parser.add_argument('--max_window', default=28, type=int)
parser.add_argument('--batch_neg', default=0, type=int)
parser.add_argument('--rand_neg', default=8, type=int)
parser.add_argument('--model', default=3, type=int)
parser.add_argument('--input_window', default=3, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition2(args.dataset, args)
[user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
num_batch = len(user_train) // args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

try:
    relation_matrix = pickle.load(open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'rb'))
except:
    relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
    pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'wb'))

sampler = WarpSampler(args, user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = TiSASRec(usernum, itemnum, itemnum, args).to(args.device)

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_uniform_(param.data)
    except:
        pass # just ignore those failed init layers

model.train() # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(args.state_dict_path))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except:
        print('failed loading state_dicts, pls check file path: ', end="")
        print(args.state_dict_path)

if args.inference_only:
    model.eval()
    t_test = evaluate(model, dataset, args)
    print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

#bce_criterion = torch.nn.BCEWithLogitsLoss()
ce_criterion = torch.nn.CrossEntropyLoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

T = 0.0
t0 = time.time()

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    if args.inference_only: break # just to decrease identition
    for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch() # tuples to ndarray
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)        
        time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)

        if args.batch_neg != 0:
          neg, logQ_inbatchn_list = mix_inbatch_neg(pos, neg, itemnum, args.batch_neg) 

        if args.model == 3:
          pos_logits, neg_logits = model(u, seq, time_matrix, pos, neg)  # all-item
        elif args.model == 4:
          all_logits, all_labels = model(u, seq, time_matrix, pos, neg) # dense-all-item 
        
        adam_optimizer.zero_grad()

        ### NEXT ITEM (BASELINE) ##############
        '''
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
        # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
        
        indices = np.where(pos != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        '''
        ### FUTURE ALL ITEM ############           
        if args.model == 3: 
          loss = 0
          for i in range(args.num_items):
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

        ### FUTURE DENSE ALL ITEM ############    
        elif args.model == 4:
          loss = 0
          for logits, labels in zip(all_logits, all_labels):
            loss += ce_criterion(logits, labels)               
          loss = loss.mean()          
        #############################################    

        for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        loss.backward()
        adam_optimizer.step()
        #print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

    if epoch % 1 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        t_test = evaluate_all(model, dataset, args)
        print('test:',  t_test)
        # t_valid = evaluate_valid(model, dataset, args)
        # print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
        #         % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

        # f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        # f.flush()
        t0 = time.time()
        model.train()

    if epoch == args.num_epochs:
        folder = args.dataset + '_' + args.train_dir
        fname = 'TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        torch.save(model.state_dict(), os.path.join(folder, fname))

f.close()
sampler.close()
print("Done")
