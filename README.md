# Sentence-Classification-with-CNN
A Chainer implementation of a Convolutional Network model for sentence classification in movie reviews dataset.

The CNN model is inspired by <a href=https://arxiv.org/abs/1408.5882> Convolutional Neural Networks for Sentence Classification</a>

<b>Requirements</b>
1. Python3
2. Chainer
3. vsmlib
4. numpy
5. Word Embeddings (It can be downloaded from https://nlp.stanford.edu/projects/glove/, the Stanford NLP group has a bunch of open source pre-trained Glove embeddings or you can use your own embeddings. Just specify the path in config.yaml)

<b>Dataset</b>
The Movie Reviews (MR) dataset (https://www.cs.cornell.edu/people/pabo/movie-review-data/) is used for this model.
The Train, dev and test sets have to be present. The path can be specified in cofig.yaml file. A small subset of the data is provided to get you started.
```
1 That was so beautiful that it can't be put into words . (POSITIVE SETENCE)
0 I do not want to go to school because I do like to study math . (NEGATIVE SENTENCE)
```
<b>Configuration parameters</b>
All the config parameters and the hyperparameters of the model can be specified in the config.yaml file.

<b>Train the model</b>
```
python3 train_cnn.py config.yaml
```
