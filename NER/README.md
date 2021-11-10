# Named Entity Recognition modelling usiong bi-directional LSTM-CNNs-CRF 

Referenced from the following code repository: 
https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial/blob/master/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb 

## Dataset
The trained model file 'glove.6B.100.txt' is too large to be pushed to github. It can be downloaded from [THIS GDRIVE](https://drive.google.com/file/d/1sG0n3-vrhCOZTrVbYTKEOcrZ2m4RSEtA/view?usp=sharing). 

The other data files are saved in the data folder: 
- eng.train
- eng.testa
- eng.testb

## Dependencies
- pytorch 

## Running scripts

The python notebooks scripts can be run in either Google Colab environment or Jupyter Notebook environment 

## Folder structure

data/       # Given data file
models/     # Trained models 
results/    # Saved csv files of model train, val, test scores

the python notebooks have been labelled respectively by its neural network model type 
eg. conv1d_1layer.ipynb is a 1 layer Conv1D model 
eg. conv1d_1layer_maxpool.ipynb is a 1 layer Conv1D model that has an additional maxpool layer 
eg. conv1d_1layer_relu.ipynb is a 1 layer Conv1D model that has an additional ReLU layer 
