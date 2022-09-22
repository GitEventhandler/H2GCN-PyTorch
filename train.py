import os
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
from utils import accuracy, set_seed, select_mask, load_dataset
from model import H2GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--k', type=int, default=2, help='number of embedding rounds')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay value')
parser.add_argument('--hidden', type=int, default=64, help='embedding output dim')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--dataset', default='cora', help='dateset name')
parser.add_argument('--gpu', type=int, default=0, help='gpu id to use while training, set -1 to use cpu')
parser.add_argument('--split-id', type=int, default=0, help='the data split to use')
args = parser.parse_args()


def train():
    model.train()
    optimizer.zero_grad()
    output = model(adj, features)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(adj, features)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test():
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    with torch.no_grad():
        output = model(adj, features)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


def main():
    begin_time = time.time()
    tolerate = 0
    best_loss = 1000
    for epoch in range(args.epochs):
        loss_train, acc_train = train()
        loss_validate, acc_validate = validate()
        if (epoch + 1) % 1 == 0:
            print(
                'Epoch {:03d}'.format(epoch + 1),
                '|| train',
                'loss : {:.3f}'.format(loss_train),
                ', accuracy : {:.2f}%'.format(acc_train * 100),
                '|| val',
                'loss : {:.3f}'.format(loss_validate),
                ', accuracy : {:.2f}%'.format(acc_validate * 100)
            )
        if loss_validate < best_loss:
            best_loss = loss_validate
            torch.save(model.state_dict(), checkpoint_path)
            tolerate = 0
        else:
            tolerate += 1
        if tolerate == args.patience:
            break
    print("Train cost : {:.2f}s".format(time.time() - begin_time))
    print("Test accuracy : {:.2f}%".format(test()[1] * 100), "on dataset", args.dataset)


if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device('cpu' if args.gpu == -1 else "cuda:%s" % args.gpu)
    features, labels, feat_dim, class_dim, adj, train_mask, val_mask, test_mask = load_dataset(
        args.dataset,
        device
    )
    checkpoint_path = utils.root + '/checkpoint/%s.pt' % args.dataset
    idx_train, idx_val, idx_test = select_mask(args.split_id, train_mask, val_mask, test_mask)
    if not os.path.exists(utils.root + '/checkpoint'):
        os.makedirs(utils.root + '/checkpoint')
    model = H2GCN(
        feat_dim=feat_dim,
        hidden_dim=args.hidden,
        class_dim=class_dim,
    ).to(device)
    optimizer = optim.Adam([{'params': model.params, 'weight_decay': args.wd}], lr=args.lr)
    main()
