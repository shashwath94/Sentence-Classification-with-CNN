# -*- coding: utf-8 -*-
import sys
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, Function, Variable
from chainer import Link, Chain


class CNN(Chain):

    def __init__(self, n_vocab, input_channel, output_channel, n_label, embed_dim, train=True):
        super(CNN, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim),  # embedding vector size
            conv3=L.Convolution2D(
                input_channel, output_channel, (3, embed_dim)),
            conv4=L.Convolution2D(
                input_channel, output_channel, (4, embed_dim)),
            conv5=L.Convolution2D(
                input_channel, output_channel, (5, embed_dim)),
            l1=L.Linear(3 * output_channel, n_label)
        )
        self.train = train

    def load_embeddings(self, glove_path, vocab):
        assert self.embed != None
        sys.stderr.write("loading word embedddings...")
        with open(glove_path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
                    self.embed.W.data[vocab[word]] = vec
        sys.stderr.write("done\n")

    def __call__(self, xs):
        self.embed.disable_update()
        xs = self.embed(xs)
        batchsize, height, width = xs.shape
        xs = F.reshape(xs, (batchsize, 1, height, width))
        conv3_xs = self.conv3(xs)
        conv4_xs = self.conv4(xs)
        conv5_xs = self.conv5(xs)
        h1 = F.max_pooling_2d(F.relu(conv3_xs), conv3_xs.shape[2])
        h2 = F.max_pooling_2d(F.relu(conv4_xs), conv4_xs.shape[2])
        h3 = F.max_pooling_2d(F.relu(conv5_xs), conv5_xs.shape[2])
        concat_layer = F.concat([h1, h2, h3], axis=1)
        with chainer.using_config('train', True):
            y = self.l1(F.dropout(F.tanh(concat_layer)))
        return y
