# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import DeepSpeech2Encoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay
LANG = 'cantonese'
base_model = Speech2Text

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 1,
  "batch_size_per_gpu": 32,

  "num_epochs": 10,

  "save_summaries_steps": 1000,
  "print_loss_steps": 10,
  "print_samples_steps": 10000,
  "eval_steps": 10000,
  "save_checkpoint_steps": 1000,
  "logdir": "experiments/ds2_medium_can-150",

  "optimizer": "Adam",
  'max_grad_norm': 1.0,
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.0002,
    "power": 0.5
  },
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
        "num_channels": 64, "padding": "SAME"
      },
      {
        "kernel_size": [11, 21], "stride": [1, 2],
        "num_channels": 96, "padding": "SAME"
      },
    ],
    "num_rnn_layers": 3,
    "rnn_cell_dim": 1024,

    "use_cudnn_rnn": True,
    "rnn_type": "cudnn_gru",
    "rnn_unidirectional": True,

    "row_conv": True,
    "row_conv_width": 8,

    "n_hidden": 2048,

    "dropout_keep_prob": 0.5,
    "activation_fn": tf.nn.relu,
    "data_format": "channels_first",
  },

  "decoder": FullyConnectedCTCDecoder,
  "decoder_params": {
    "use_language_model": False,
    # "infer_logits_to_pickle": True,
    # params for decoding the sequence with language model
    "beam_width": 512,
    "alpha": 2.0,
    "beta": 1.0,

    "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
    "lm_path": "/home/lokhiufung/data/{}/lm/6-gram.binary".format(LANG),
    # "trie_path": "language_model/trie.binary",
    "alphabet_config_path": "/home/lokhiufung/data/{}/vocab.txt".format(LANG),
  },
  "loss": CTCLoss,
  "loss_params": {},
}

train_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 96,
    "input_type": "spectrogram",
    "augmentation": {'time_stretch_ratio': 0.05,
                     'noise_level_min': -90,
                     'noise_level_max': -60},
    "vocab_file": "/home/lokhiufung/data/{}/vocab.txt".format(LANG),
    "dataset_files": [
      "/home/lokhiufung/data/{}/train.csv".format(LANG),
    ],
    "max_duration": 11.7,
    "shuffle": True,
  },
}

eval_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 96,
    "input_type": "spectrogram",
    "vocab_file": "/home/lokhiufung/data/{}/vocab.txt".format(LANG),
    "dataset_files": [
      "/home/lokhiufung/data/{}/dev.csv".format(LANG),
    ],
    "shuffle": False,
  },
}

infer_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 96,
    "input_type": "spectrogram",
    "vocab_file": "/home/lokhiufung/data/{}/vocab.txt".format(LANG),
    "dataset_files": [
      "/home/lokhiufung/data/{}/test_.csv".format(LANG),
    ],
    "shuffle": False,
  },
}

interactive_infer_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 96,
    "input_type": "spectrogram",
    "vocab_file": "/home/lokhiufung/data/{}/vocab.txt".format(LANG),
    "dataset_files": [],
    "shuffle": False,
  },
}
