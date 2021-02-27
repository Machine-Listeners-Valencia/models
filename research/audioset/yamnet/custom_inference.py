"""Inference demo for YAMNet."""
from __future__ import division, print_function

from tests.config import path2yamnetmap, path2yamnet
import sys
sys.path.insert(0, path2yamnet)

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import params as yamnet_params
import yamnet as yamnet_model


def inference(path2model, file_name):
    # assert argv, 'Usage: inference.py <wav file> <wav file> ...'

    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights(path2model)
    yamnet_classes = yamnet_model.class_names(path2yamnetmap)

    # Decode the WAV file.
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    # Predict YAMNet classes.
    scores, embeddings, spectrogram = yamnet(waveform)

    print('------ Embedding type ------')
    print(type(embeddings))

    embeddings_numpy = embeddings.numpy()
    print('------ Embedding type ------')
    print(type(embeddings_numpy))

    print('------ Embedding shape ------')
    print(embeddings_numpy.shape)

    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top5_i = np.argsort(prediction)[::-1][:5]
    print(file_name, ':\n' +
          '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                    for i in top5_i))
