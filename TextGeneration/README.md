# Word-level language modeling (FNN, LSTM, Transformer)

We have already included the dataset here in the `data/wikitext-2 folder`.

## Dependencies

1. Python libraries

The following python packages can be installed by:
```
pip install -r requirements.txt
```

2. Preprocessed data tensors

The preprocessing of data for FNN model might take some time, therefore the saved preprocessed data tensors can be obtained by:
```
unzip data_tensors.zip
```

3. Pre-trained models

The trained model files are too large to be pushed to github, only the default `FNN_adam_tied.dat` model can be found in the `models/` folder. For the rest of the models, they can be downloaded from [THIS GDRIVE](https://drive.google.com/drive/folders/1lTD7Hf5e-p-cNwkZy-egybdRKcE3aRn7). They shoudl be placed into the `models/` folder.

## Running scripts

The scripts are suitable to run on both CPU and GPU. However, it is advised to train the model using GPU, using the `--cuda` flag.

### Training script: `main.py`

Default params:

`--lr 4e-4 --epochs 20 --model FNN --emsize 200 --nhid 200 --ngram_size 8 --batch_size 512 --optim adam`

*Note: For --tied models, --emsize must be same as --nhid*

```bash
# Train models (FNN)
python main.py --cuda --tied                      # FNN_adam_tied - FNN tied model with Adam
python main.py --cuda                             # FNN_adam_non-tied - FNN non-tied model with Adam
python main.py --cuda --tied --optim rmsprop      # FNN_rmsprop_tied - FNN tied model with RMSprop

# Train models (LSTM)
python main.py --cuda --tied --model LSTM         # LSTM_adam_tied - LSTM tied model with Adam

# Train models (Transformer)
python main.py --cuda --tied --model Transformer  # Transformer_adam_tied - Transformer tied model with Adam
```
### Inference script: `inference.py`
User input 7 words, output the most probably next word.

```bash
# Predict next word of a 7-word input sentence using trained FNN_adam_tied model
python inference.py  --model_path ./models/FNN_adam_tied.dat  --input "today is a good day and I"

# Other models to try:
python inference.py  --model_path ./models/FNN_adam_not-tied.dat  --input "today is a good day and I"
python inference.py  --model_path ./models/FNN_rmsprop_tied.dat  --input "today is a good day and I"
```

### Generate script: `generate.py`
Randomly extracts a 7-word sentence from the full corpus as input to the model and generates 20 words.

```bash
# Generate text (20 new words)
python generate.py  --model_path ./models/FNN_adam_tied.dat --words 20

```




The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         Type of model (FNN, LSTM, Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --lr LR               initial learning rate
  --optim OPTIMIZER     Optimizer (adam, rmsprop)
  --clip GRADIENT       Gradient clipping
  --dropout DROPOUT     Dropout probability
  --epochs EPOCHS       upper epoch limit
  --ngram_size NGRAM    Size of ngram
  --seed SEED           random seed
  --cuda                use CUDA
  --tied                tie the word embedding and softmax weights
  --batch_size N        batch size
  --log-interval        Interval to log training metrics
  --nlayers             Number of RNN layers
  --nhead               Number of Transformer heads

```

## Folder structure

```bash
data/       # Given data file
generated/  # Generated text
logs/       # Model training logs
models/     # Model files (.dat)

data.py
generate.py
inference.py
main.py
model.py
```
