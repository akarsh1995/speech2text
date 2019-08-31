#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np

import sys

from deepspeech import Model, printVersions
from timeit import default_timer as timer
from config import Config
from deepspeech_model.record_audio import RecordOrRead
c = Config()


# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75

# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        printVersions()
        exit(0)


def main():

    print('Loading model from file {}'.format(c.model), file=sys.stderr)
    model_load_start = timer()
    ds = Model(c.model, N_FEATURES, N_CONTEXT, c.alphabet, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    if c.lm and c.trie:
        print('Loading language model from files {} {}'.format(c.lm, c.trie), file=sys.stderr)
        lm_load_start = timer()
        ds.enableDecoderWithLM(c.alphabet, c.lm, c.trie, LM_ALPHA, LM_BETA)
        lm_load_end = timer() - lm_load_start
        print('Loaded language model in {:.3}s.'.format(lm_load_end), file=sys.stderr)
    r = RecordOrRead()
    audio = r.record_audio()
    fs = r.SAMPLE_RATE
    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    if c.extended:
        print(metadata_to_string(ds.sttWithMetadata(audio, fs)))
    else:
        print(ds.stt(audio, fs))
    inference_end = timer() - inference_start
    print('Inference took %0.3fs' % inference_end, file=sys.stderr)

