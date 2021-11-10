import argparse

import torch
import random
import model
import data as data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model_path', type=str, default='./models/FNN_adam_tied.dat',
                    help='model (.dat file) to use')
parser.add_argument('--gentext', type=str, default='./generated/FNN_adam_tied.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='20',
                    help='number of words to generate')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
parser.add_argument('--ngram_size', type=int, default=8, metavar='N',
                    help='ngram size')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--tied', action='store_true',
                    help='Share embedding weights for input and output')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

corpus = data.Corpus(args.data)
# combine the train, valid, test to get a full corpus
full_corpus = torch.cat((corpus.train, corpus.valid, corpus.test))
ntokens = len(corpus.dictionary)

if args.model_path.split('.')[-2].split('_')[-1] == 'tied': args.tied = True

best_model = model.FNNmodel(ntokens, args.emsize, args.ngram_size - 1, args.nhid, args.tied).to(device)
best_model.load_state_dict(torch.load(args.model_path, map_location=device))


best_model.eval()
with open(args.gentext, 'w') as gentext:
    #randomly get a starting word from the full corpus
    start = random.randint(0, len(full_corpus)-7)
    word_gram = full_corpus[start:start+7]
    generated_text=word_gram.to(device)

    for i in range(args.words):
        with torch.no_grad():
            # get the next word
            output = best_model(generated_text[-7:])
            # get the idx of the word with highest probability
            word_idx = torch.argmax(output, dim=1)
            # combine together
            generated_text = torch.cat((generated_text,word_idx))

    # convert idx to words
    sent = [corpus.dictionary.idx2word[i] for i in generated_text]
    output_sent = ' '.join(sent)
    gentext.write(output_sent)
    sent = sent[:7] + ['>>> generated text >>>'] + sent[7:]
    print("###",' '.join(sent))
    print('-'*50)
    orig = [corpus.dictionary.idx2word[i] for i in full_corpus[start:start+len(sent)-7]]
    print("$$$",' '.join(orig))
