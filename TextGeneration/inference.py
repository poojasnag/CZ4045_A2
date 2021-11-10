import argparse
import torch
import data
import model


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--model_path', type=str, default='./models/FNN_adam_tied.dat',
                        help='location of the best model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--ngram_size', type=int, default=8, metavar='N',
                        help='ngram size')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--input', type=str, default='i like fruits and they include some',
                        help='input string for inference')
    parser.add_argument('--tied', action='store_true',
                        help='Share embedding weights for input and output')
    parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
    return parser


def get_id_of_word(word):
    unknown_word_id = dict_obj.word2idx['<unk>']
    return dict_obj.word2idx.get(word, unknown_word_id)
    # return dict_obj.word2idx.get(word)

def preprocess_test(text):
  # hi my name is cammy i like
  x_extract = [get_id_of_word(word.lower()) for word in text.split()]
  return torch.LongTensor(x_extract)

def get_preds(best_model, input_tensors, dict_obj):
    prob = best_model(input_tensors)
    prob1 = 10**prob[0]

    prob_list = []
    for idx, i in enumerate(list(prob1)):
        prob_list.append((float(i), idx))
    prob_list.sort()
    top3 = prob_list[::-1][:10]
    return [dict_obj.idx2word[idx] for _, idx in top3]


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(' ########## GENERATE TEXT ########## ')
    corpus = data.Corpus(args.data)
    dict_obj = corpus.dictionary
    ntokens = len(corpus.dictionary)

    best_model_path = args.model_path

    if best_model_path.split('.')[-2].split('_')[-1] == 'tied':
        args.tied = True

    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    input_tensors = preprocess_test(args.input).to(device)

    # ---------------------- Loading Best Model -------------------
    best_model = model.FNNmodel(ntokens, args.emsize, args.ngram_size - 1, args.nhid, args.tied).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    preds = get_preds(best_model, input_tensors, dict_obj)

    print("Next word: ", preds[0])
