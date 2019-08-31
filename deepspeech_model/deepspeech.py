#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import sys
from deepspeech import Model
from timeit import default_timer as timer
from config import Config
from deepspeech_model.record_audio import RecordOrRead
c = Config()


def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)


def main():

    print('Loading model from file {}'.format(c.model), file=sys.stderr)
    model_load_start = timer()
    ds = Model(c.model, c.N_FEATURES, c.N_CONTEXT, c.alphabet, c.BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    if c.lm and c.trie:
        print('Loading language model from files {} {}'.format(c.lm, c.trie), file=sys.stderr)
        lm_load_start = timer()
        ds.enableDecoderWithLM(c.alphabet, c.lm, c.trie, c.LM_ALPHA, c.LM_BETA)
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

