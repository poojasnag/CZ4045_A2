# coding: utf-8
import argparse
import time
import math
import multiprocessing
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import torch.nn as nn
import torch.onnx
from tqdm import tqdm

import data
import model

GPU = 0

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='FNN',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer or FNN)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='initial learning rate')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optimizer (adam, rmsprop, sgd)')
    parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--ngram_size', type=int, default=8, metavar='N',
                        help='ngram size')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--tied', action='store_true',
                        help='Share embedding weights for input and output')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='data batch size')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

    return parser


def batchify(data, bsz):
    """
    Data preprocessing method to get output data with bsz columns
    """
    if args.model == "FNN":
        train_data = []
        for i in tqdm(range(len(data) - bsz + 1)):
            ngram = [data[i+j] for j in range(bsz)]

            train_data.append(ngram)
        return torch.LongTensor(train_data)
    else:
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()

        return data


def load_data(train_file_path, val_file_path, test_file_path):
    """
    Get respective train/val/test data
    """
    if not os.path.isdir('./data_tensors/'):
        os.mkdir('./data_tensors/')

    if os.path.isfile(train_file_path):
        print('using saved train data')
        train_data = torch.load(train_file_path)
    else:
        print('converting training data...')
        train_data = batchify(corpus.train, args.ngram_size)
        torch.save(train_data, train_file_path)

    if os.path.isfile(val_file_path):
        print('using saved val data')
        val_data = torch.load(val_file_path)
    else:
        print('converting val data...')
        val_data = batchify(corpus.valid, args.ngram_size)
        torch.save(val_data, val_file_path)

    if os.path.isfile(test_file_path):
        print('using saved test data')
        test_data = torch.load(test_file_path)
    else:
        print('converting test data...')
        test_data = batchify(corpus.test, args.ngram_size)
        torch.save(test_data, test_file_path)

    return train_data, val_data, test_data

def get_accuracy_from_log_probs(log_probs, labels):
    """
    Compares output tensor of log probability with the ground truth label to get accuracy
    """
    probs = torch.exp(log_probs)
    predicted_label = torch.argmax(probs, dim=1)
    acc = (predicted_label == labels).float().mean()
    return acc

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    """
    Transforms input data into predictor and target variables
    """
    seq_len = min(args.batch_size, len(source) - 1 - i)
    if args.model == "FNN":
        # Generate batches based on sliding window
        data = source[i:i+seq_len]
        target = data[:, -1]
        data = data[:, :-1]
        return data, target
    else:
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)

        return data, target


def evaluate(model, loss_function, data_source):
    """
    helper function to evaluate model on dev data
    """
    model.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    if args.model != 'Transformer' and args.model != 'FNN':
        hidden = model.init_hidden(args.ngram_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.batch_size):
            data, targets = get_batch(data_source, i)
            data = data.to(device)
            targets = targets.to(device)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            elif args.model == "FNN":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)

            mean_loss += loss_function(output, targets).item()
            mean_acc += get_accuracy_from_log_probs(output, targets)
            count += 1


    return mean_acc / count, mean_loss / count


def train(args, model, train_data, val_data):
    import sys

    result_df = pd.DataFrame()
    loss_function = nn.NLLLoss()

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    if args.model != 'Transformer' and args.model != 'FNN':
        hidden = model.init_hidden(args.ngram_size)
     # ------------------------- TRAIN & SAVE MODEL ------------------------
    best_perplexity = sys.maxsize
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0,verbose=True)

    try:
        for epoch in range(1, args.epochs+1):
            model.train()
            st = time.time()
            epoch_start_time = time.time()
            print("\n--- Training model Epoch: {} ---".format(epoch))
            total_loss = 0
            total_acc = 0


            # for it, data_tensor in enumerate(train_loader):
            for batch, i in enumerate(range(0, train_data.size(0) - 1, args.batch_size)):
                data, targets = get_batch(train_data, i)
                data = data.to(device)
                targets = targets.to(device)

                # zero out the gradients from the old instance
                model.zero_grad()

                # get log probabilities over next words
                if args.model == 'Transformer':
                    output = model(data)
                    output = output.view(-1, ntokens)
                elif args.model == 'FNN':
                    output = model(data)
                    output = output.view(-1, ntokens)
                else:
                    hidden = repackage_hidden(hidden)
                    output, hidden = model(data, hidden)

                # calculate current accuracy
                acc = get_accuracy_from_log_probs(output, targets)
                total_acc += acc
                # compute loss function
                loss = loss_function(output, targets)

                total_loss += loss.item()
                # backward pass and update gradient
                loss.backward()
                optimizer.step()


                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                for p in model.parameters():
                    p.data.add_(p.grad, alpha=-args.lr)

                if batch % args.log_interval == 0 and batch > 0:
                    cur_loss = total_loss / args.log_interval
                    time_elapsed = time.time() - st

                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2e} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.batch_size, optimizer.param_groups[0]['lr'],
                    time_elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

                    st = time.time()
                    total_loss, total_acc = 0, 0

            print("\n--- Evaluating model on dev data ---")
            dev_acc, dev_loss = evaluate(model, loss_function, val_data)
            dev_perplexity = math.exp(dev_loss)

            epoch_end_time = time.time()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (epoch_end_time - epoch_start_time),
                                            dev_loss, math.exp(dev_loss)))
            print('-' * 89)
            scheduler.step(dev_loss)

            if dev_perplexity < best_perplexity:
                if not os.path.exists('./models/'):
                    os.makedirs('./models/')

                print("Best development perplexity improved from {:8.2f} to {:8.2f}, saving model to {}".format(best_perplexity, dev_perplexity, best_model_path))
                best_perplexity = dev_perplexity

                torch.save(model.state_dict(), best_model_path)

            # Append results to pandas DataFrame for analysis of validation results
            result_df = result_df.append({
                "epoch": epoch,
                "train_perplexity":  math.exp(cur_loss),
                "val_perplexity": math.exp(dev_loss),
                "val_loss": dev_loss,
                "time": epoch_end_time - epoch_start_time,
            }, ignore_index=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

        if best_perplexity == sys.maxsize:
            exit()

    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')

    csv_filename = f"./logs/{model_name}.csv"
    result_df.to_csv(csv_filename)


def test(model, test_data):
    # Run on test data.
    loss_function = nn.NLLLoss()
    test_acc, test_loss = evaluate(model, loss_function, test_data)
    print('=' * 89)
    print('| Test data | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()



    print(' ########## TRAIN MODEL ########## ')
    print(f"Model: {args.model}, Ngram size: {args.ngram_size}, Tied Weights: {args.tied}, CUDA: {args.cuda}, nhid: {args.nhid}, emsize: {args.emsize}, Epochs: {args.epochs}, LR: {args.lr}, Optimizer: {args.optim}")
    model_name = f"{args.model}_{args.optim}_tied" if args.tied else f"{args.model}_{args.optim}_not-tied"
    best_model_path = f"./models/{model_name}.dat"

    if args.model != "FNN":
        args.ngram_size, args.batch_size = args.batch_size, args.ngram_size
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")
    available_workers = multiprocessing.cpu_count()

    # Load data
    corpus = data.Corpus(args.data)

    train_file_path = './data_tensors/train_data_fnn.pt' if args.model == "FNN" else './data_tensors/train_data.pt'
    val_file_path = './data_tensors/val_data_fnn.pt' if args.model == "FNN" else './data_tensors/val_data.pt'
    test_file_path = './data_tensors/test_data_fnn.pt' if args.model == "FNN" else './data_tensors/test_data.pt'

    train_data, val_data, test_data = load_data(train_file_path, val_file_path, test_file_path)
    # Build the model
    ntokens = len(corpus.dictionary)

    if args.model == 'Transformer':
        model_ = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
        print('Transformer model created')
    elif args.model == "FNN":
        model_ = model.FNNmodel(ntokens, args.emsize, args.ngram_size - 1, args.nhid, args.tied).to(device)
        print('FNN model created')
    else:
        model_ = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
        print(f'RNN model created')

    print("Total Trainable params: ", sum(p.numel() for p in model_.parameters() if p.requires_grad))

    train(args, model_, train_data, val_data)

    print(f'loading best model from {best_model_path}')
    model_.load_state_dict(torch.load(best_model_path, map_location=device))

    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model_.rnn.flatten_parameters()

    test(model_, test_data)




