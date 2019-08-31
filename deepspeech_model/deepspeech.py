#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from deepspeech import Model
from timeit import default_timer as timer
from config import Config
from deepspeech_model.record_audio import RecordOrRead

c = Config()


def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)


def get_sound(file_path):
    if file_path:
        r = RecordOrRead(file_path)
        audio = r.read_file()
    else:
        r = RecordOrRead()
        audio = r.record_audio()
    return audio, r.SAMPLE_RATE


class SpeechToText:

    def __init__(self):
        self.c = Config()
        self.model = self.get_model()

    def get_model(self):
        print('Loading model from file {}'.format(self.c.model), file=sys.stderr)
        model_load_start = timer()
        model = Model(self.c.model, self.c.N_FEATURES, self.c.N_CONTEXT, self.c.alphabet, self.c.BEAM_WIDTH)
        model_load_end = timer() - model_load_start
        print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

        if self.c.lm and self.c.trie:
            print('Loading language model from files {} {}'.format(self.c.lm, self.c.trie), file=sys.stderr)
            lm_load_start = timer()
            model.enableDecoderWithLM(self.c.alphabet, self.c.lm, self.c.trie, self.c.LM_ALPHA, self.c.LM_BETA)
            lm_load_end = timer() - lm_load_start
            print('Loaded language model in {:.3}s.'.format(lm_load_end), file=sys.stderr)
        return model

    def predict(self, file_path=None):
        audio, fs = get_sound(file_path=file_path)
        print('Running inference.', file=sys.stderr)
        inference_start = timer()
        if c.extended:
            text = metadata_to_string(self.model.sttWithMetadata(audio, fs))
        else:
            text = self.model.stt(audio, fs)
        inference_end = timer() - inference_start
        print('Inference took %0.3fs' % inference_end, file=sys.stderr)
        return text
