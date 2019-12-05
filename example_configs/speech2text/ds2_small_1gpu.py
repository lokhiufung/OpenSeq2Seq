# pylint: skip-file
import tensorflow as tf

from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.encoders import DeepSpeech2Encoder
from open_seq2seq.losses import CTCLoss
from open_seq2seq.models import Speech2Text
from open_seq2seq.optimizers.lr_policies import exp_decay


base_model = Speech2Text

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 200,

  "num_gpus": 1,
  "batch_size_per_gpu": 32,

  "save_summaries_steps": 100,
  "print_loss_steps": 10,
  "print_samples_steps": 5000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 1000,
  "logdir": "experiments/ds2_small_man",

  "optimizer": "Adam",
  "optimizer_params": {},
  'max_grad_norm': 1.0,
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 0.0001,
    "begin_decay_at": 0,
    "decay_steps": 5000,
    "decay_rate": 0.9,
    "use_staircase_decay": True,
    "min_lr": 0.0,
  },
  "dtype": tf.float32,
  # weight decay
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0005
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": DeepSpeech2Encoder,
  "encoder_params": {
    "conv_layers": [
      {
        "kernel_size": [11, 41], "stride": [2, 2],
        "num_channels": 32, "padding": "SAME"
      },
      {
        "kernel_size": [11, 21], "stride": [1, 2],
        "num_channels": 32, "padding": "SAME"
      }
    ],
    "num_rnn_layers": 2,
    "rnn_cell_dim": 512,

    "use_cudnn_rnn": True,
    "rnn_type": "cudnn_gru",
    "rnn_unidirectional": False,

    "row_conv": False,

    "n_hidden": 1024,

    "dropout_keep_prob": 0.5,
    "activation_fn": tf.nn.relu,
    "data_format": "channels_first",
  },

  "decoder": FullyConnectedCTCDecoder,
  "decoder_params": {
    "use_language_model": False,

    # params for decoding the sequence with language model
    "beam_width": 512,
    "alpha": 2.0,
    "beta": 1.0,

    "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
    "lm_path": "/home/lokhiufung/data/mandarin/lm/4-gram.binary",
    # "trie_path": "language_model/trie.binary",
    "alphabet_config_path": "/home/lokhiufung/data/mandarin/vocab.txt",
  },
  "loss": CTCLoss,
  "loss_params": {},
}

train_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 96,
    "input_type": "spectrogram",
    "augmentation": {
      'time_stretch_ratio': 0.05,
      'noise_level_min': -90,
      'noise_level_max': -60
    },
    "vocab_file": "/home/lokhiufung/data/mandarin/vocab.txt",
    "dataset_files": [
      "/home/lokhiufung/data/mandarin/train.csv",
    ],
    "max_duration": 13.166,
    "shuffle": True,
  },
}

eval_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 96,
    "input_type": "spectrogram",
    "vocab_file": "/home/lokhiufung/data/mandarin/vocab.txt",
    "dataset_files": [
      "/home/lokhiufung/data/mandarin/dev.csv",
    ],
    "shuffle": False,
  },
}
