import logging
import argparse
import time
import numpy as np
from numpy import random
import torch
import torch.nn.functional as F
import dgl
from utils import EarlyStopping
from model import BANADA
from sklearn.metrics import f1_score,classification_report,roc_auc_score
from dgl.data.utils import load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from sklearn.model_selection import train_test_split
import os

"""
	Training BANADA
	Paper: Who Are the Evil Backstage Manipulators: Boosting Graph Attention Networks against Deep Fraudsters
"""

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits,_, attention = model(features)
        logits = torch.reshape(logits, [logits.shape[0], -1])
        logits = logits[mask]
        labels = labels[mask]
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss = loss_fcn(logits, labels)
        return accuracy(logits, labels), loss, logits


def gen_mask(g, train_rate, val_rate):
    labels = g.ndata['label']
    g.ndata['label'] = labels.long()
    labels = np.array(labels)
    n_nodes = len(labels)
    index=list(range(n_nodes))
    train_idx, val_test_idx, _, y_validate_test = train_test_split(index, labels, stratify=labels, train_size=train_rate,test_size=1-train_rate,
                                                 random_state=2, shuffle=True)
    val_idx, test_idx, _, _ = train_test_split(val_test_idx,y_validate_test, train_size=val_rate/(1-train_rate), test_size=1-val_rate/(1-train_rate),
                                                     random_state=2, shuffle=True)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    return g,train_idx


parser = argparse.ArgumentParser(description='BANADA')
parser.add_argument('--dataset', type=str, default='Sichuan', help='Sichuan,BUPT')
parser.add_argument("--dropout", type=float, default=0.3, help="dropout probability")
parser.add_argument("--adj_dropout", type=float, default=0.3, help="mixed dropout for adj")
parser.add_argument('--layers', type=int, default=1, help='Number of Basic-model layers.')
parser.add_argument("--num_layers", type=int, default=1, help="number of attention-hidden layers")
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units. ')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer(all layers).')
parser.add_argument('--reg', type=float, default=5e-3, help='Weight decay on the 1st layer.')
parser.add_argument("--epochs", type=int, default=300, help="number of training epochs")
parser.add_argument('--patience', type=int, default=200, help='patience in early stopping')
parser.add_argument('--num_heads', type=int, default=1, help='number of hidden attention heads')
parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
parser.add_argument("--in_drop", type=float, default=0.1, help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=0.1, help="attention dropout")
parser.add_argument('--early_stop', action='store_true', default=True,
                    help="indicates whether to use early stop or not")
parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
parser.add_argument('--negative_slope', type=float, default=0.2, help="the negative slope of leaky relu")
parser.add_argument('--print_interval', type=int, default=10, help="the interval of printing in training")
parser.add_argument('--seed', type=int, default=42, help="seed for our system")
parser.add_argument('--att_loss_weight', type=float, default=0.5, help="attention loss weight")
parser.add_argument('--attention_weight', type=float, default=0.7, help='External Attention coefficient.')
parser.add_argument('--feature_weight', type=float, default=0.4, help='Feature adjust coefficient about attention.')
parser.add_argument('--train_size', type=float, default=0.2, help='train size.')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

setup_seed(args.seed)

# logging configuration
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=20,
    # filename=log_name+'.log',
    # filemode='a'
    )

# load data and preprocessing
if args.dataset == 'Sichuan':
    dataset, _ = load_graphs("./data/Sichuan_tele.bin")
    n_classes = load_info("./data/Sichuan_tele.pkl")['num_classes']
    graph = dataset[0]
    g,train_idx = gen_mask(graph, args.train_size, 0.2)
elif args.dataset == 'BUPT':
    dataset, _ = load_graphs("./data/BUPT_tele.bin")
    n_classes = load_info("./data/BUPT_tele.pkl")['num_classes']
    graph = dataset[0]
    g,train_idx = gen_mask(graph, args.train_size, 0.2)

for e in g.etypes:
    g = g.int().to(device)
    dgl.remove_self_loop(g,etype=e)
    dgl.add_self_loop(g,etype=e)

features = g.ndata['feat'].float()
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
num_feats = features.shape[1]
num_edges = g.num_edges()

##training
# create model
heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
model = BANADA(g,
            args.num_layers,
            num_feats,
            args.hid,
            n_classes,
            heads,
            F.elu,
            args.dropout,
            args.adj_dropout,
            args.in_drop,
            args.attn_drop,
            args.negative_slope,
            args.residual
            )
# print(model)
if args.early_stop:
    stopper = EarlyStopping(args.patience)
model.to(device)
loss_fcn = torch.nn.CrossEntropyLoss()

# train
start_time = time.time()
last_time = start_time

# initialize node weights in Adaboost
sample_weights = torch.ones(g.adj().shape[0])
sample_weights = sample_weights[train_mask]
sample_weights = sample_weights / sample_weights.sum()
sample_weights = sample_weights.to(device)
results = torch.zeros(g.adj().shape[0], n_classes).to(device)
ALL_epochs = 0

for layer in range(args.layers):
    # free cache
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    logging.info(f"|This is the {layer + 1}th layer!")
    stopper.best_epoch = None
    stopper.best_score = None
    stopper.early_stop = False
    stopper.counter = 0

    # use optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits,logits_GAT, _ = model(features)
        logits = torch.reshape(logits, [logits.shape[0], -1])
        logits_GAT = torch.reshape(logits_GAT, [logits.shape[0], -1])
        # Eq.(16) in the paper
        loss = F.nll_loss(F.log_softmax(logits[train_mask], 1), labels[train_mask], reduction='none')\
        +args.att_loss_weight*F.nll_loss(F.log_softmax(logits_GAT[train_mask], 1), labels[train_mask], reduction='none')
        loss = loss * sample_weights
        loss = loss.sum()
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        train_loss = loss.item() * 1.0

        # print loss and accuracy during train process
        with torch.no_grad():
            val_acc, val_loss, val_logits = evaluate(model, features, labels, val_mask)
        if epoch % args.print_interval == 0:
            duration = time.time() - last_time  # each interval including training and early-stopping
            last_time = time.time()
            if args.early_stop:
                logging.info(f"Epoch {epoch}: "
                             f"Train loss = {train_loss:.2f}, "
                             f"Train acc = {train_acc * 100:.1f}, "
                             f"Validation loss = {val_loss:.2f}, "
                             f"Validation acc = {val_acc * 100:.1f} "
                             f"({duration:.3f} sec)")
            else:
                logging.info(f"Epoch {epoch}: "
                             f"Train loss = {train_loss:.2f}, "
                             f"train acc = {train_acc * 100:.1f}, "
                             f"({duration:.3f} sec)")

        # save model parameters with early stopping
        if args.early_stop:
            if stopper.step(val_acc, model, epoch):
                break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # SAMME.R algorithm
    runtime = time.time() - start_time
    logging.log(21,
                f"Last epoch: {epoch}, best epoch: {stopper.best_epoch},best acc in {layer + 1}th layer:{stopper.best_score * 100:.2f}， ({runtime:.3f} sec)")
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    with torch.no_grad():
        ada_use_h,_, attention = model(features)

    # attention matrix
    attention = torch.reshape(attention, [attention.shape[0], -1])
    # Eq.(12) in the paper
    output_logp = torch.log(F.softmax(ada_use_h, dim=1))
    # Eq.(13) in the paper
    h = (n_classes - 1) * (output_logp - torch.mean(output_logp, dim=1).view(-1, 1))
    # Eq.(15) in the paper
    results += h

    # update node weights
    # Eq.(14) in the paper
    temp = F.nll_loss(F.log_softmax(ada_use_h[train_mask], 1), labels[train_mask], reduction='none')
    weight = sample_weights * torch.exp(-(n_classes - 1) / n_classes  * temp)
    weight = weight / weight.sum()
    sample_weights = weight.detach()

    # compute sparse attention matrix
    row=g.edges()[0].cpu().detach().numpy()
    column=g.edges()[1].cpu().detach().numpy()
    data=attention.cpu().detach().numpy().T.squeeze()
    shape=[g.ndata['feat'].shape[0], g.ndata['feat'].shape[0]]
    attention=torch.sparse_coo_tensor(torch.tensor([row,column]).to(device), torch.tensor(data).to(device),shape)
    # update features
    # Eq.(7) and Eq.(9) in the paper
    features = torch.sparse.mm(args.attention_weight * attention, args.feature_weight * features).detach()

# final result evaluation
runtime = time.time() - start_time
val_h = torch.argmax(results[val_mask], dim=1)
val_acc = torch.sum(val_h == labels[val_mask]) * 1.0 / len(labels[val_mask])
val_f1 = f1_score(labels[val_mask].cpu(),val_h.cpu(),  average='weighted')
logging.log(22,f"Validation weighted F1: {val_f1 * 100:.1f}%   Validation accuracy: {val_acc * 100:.1f}%")

test_h = torch.argmax(results[test_mask], dim=1)
test_acc = torch.sum(test_h == labels[test_mask]) * 1.0 / len(labels[test_mask])
test_f1 = f1_score(labels[test_mask].cpu(),test_h.cpu(),  average='macro')
logging.log(23,f"Test macro F1: {test_f1 * 100:.1f}%   Test accuracy: {test_acc * 100:.1f}%")


if np.isnan(results[test_mask].cpu().detach().numpy()).any() == True:
    results[test_mask]=torch.tensor(np.nan_to_num(results[test_mask].cpu().detach().numpy())).to(device)
    test_h = torch.argmax(results[test_mask], dim=1)
else:
    pass

# calculate macro AUC
if n_classes==2:
    test_auc = roc_auc_score(labels[test_mask].cpu(), torch.softmax(results[test_mask].cpu(), dim=1)[:,1],average='macro')
else:
    test_auc = roc_auc_score(labels[test_mask].cpu(), torch.softmax(results[test_mask].cpu(), dim=1),average='macro', multi_class='ovo')
logging.log(23,f"Test macro AUC: {test_auc * 100:.2f}% ")

# print report
target_names=['{}'.format(i) for i in range(n_classes)]
report = classification_report(labels[test_mask].cpu().detach().numpy(), test_h.cpu().detach().numpy(), target_names=target_names, digits=4)
logging.log(23,f"\nReport=:\n {report}")
