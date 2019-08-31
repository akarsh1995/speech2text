import os
model_dir = 'models'


class Config:
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

    model = os.path.join(model_dir, 'output_graph.pbmm')
    alphabet = os.path.join(model_dir, 'alphabet.txt')
    lm = os.path.join(model_dir, 'lm.binary')
    trie = os.path.join(model_dir, 'trie')
    extended = False
