# coding: utf-8
import numpy as np
from collections import defaultdict
import six
import sys
import yaml
import chainer
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.training import extensions
import argparse

from chainer import training
from CNN import CNN, DataProcessor, concat_examples


def main(options):
    
    #load the config params
    gpu = options['gpu']
    data_path = options['path_dataset']
    embeddings_path = options['path_vectors']
    n_epoch = options['epochs']
    batchsize = options['batchsize']
    test = options['test']
    embed_dim = options['embed_dim']

    #load the data
    data_processor = DataProcessor(data_path, test)
    data_processor.prepare_dataset()
    train_data = data_processor.train_data
    dev_data = data_processor.dev_data

    vocab = data_processor.vocab
    cnn = CNN(n_vocab=len(vocab), input_channel=1,
                  output_channel=10, n_label=2, embed_dim=embed_dim)
    cnn.load_embeddings(embeddings_path, data_processor.vocab)
    model = L.Classifier(cnn)
    if gpu >= 0:
        model.to_gpu()
    
    #steup the optimizer
    optimizer = O.Adam()
    optimizer.setup(model)
    #print(train_data)

    train_iter = chainer.iterators.SerialIterator(train_data, batchsize)
    dev_iter = chainer.iterators.SerialIterator(dev_data, batchsize,repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, converter=concat_examples, device=gpu)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'))

    # Evaluation

    eval_model = model.copy()
    eval_model.predictor.train = False
    trainer.extend(extensions.Evaluator(dev_iter, model, device=gpu, converter=concat_examples))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # take a shapshot when the model achieves highest accuracy in dev set
    #trainer.extend(extensions.snapshot_object(
     #   model, 'model_epoch_{.updater.epoch}',
      #  savefun=chainer.training.triggers.MaxValueTrigger('validation/main/accuracy', trigger=(1, 'epoch'))))
    
    trainer.run()


if __name__ == '__main__':
    
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,
                        default=-1, help='negative value indicates CPU')
    parser.add_argument('--epoch', dest='epoch', type=int,
                        default=5, help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=32, help='learning minibatch size')
    parser.add_argument('--embedding_path', dest='glove_path', type=str,
                        default="/home/shashwath/ldt/Li-wiki201308_filtered/structured_deps/cbow/100d/vectors.txt", help='Pretrained glove vector')
    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.set_defaults(test=False)

    args=parser.parse_args()

    '''
    options = {}
    if len(sys.argv) > 1:
        path_config = sys.argv[1]
    else:
        print("usage: python3 -m vsmlib.benchmarls.sequence_labeling.sentence_classification <config file>")
        print("config file example can be found at ")
        print("https://github.com/undertherain/vsmlib/blob/master/vsmlib/benchmarks/sequence_labeling/sequence_labeling/config.yaml")
        
    with open(path_config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    
    #options["path_data"] = cfg["path_data"]
    options["path_vectors"] = cfg["path_vectors"]
    options["path_dataset"] = cfg["path_dataset"]
    options["gpu"] = cfg["gpu"]
    options["epochs"] = cfg["epochs"]
    options["batchsize"] = cfg["batch_size"]
    options["test"] = cfg["test"]
    options["embed_dim"] = cfg["embed_dim"]

    main(options)
