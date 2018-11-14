import os

import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

from hparams import hparams
from models import create_model
from util import audio


class Synthesizer:
    def load(self, checkpoint_path, vgg19_path, model_name='tacotron'):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.float32, [1, hparams.image_dim, hparams.image_dim, 3], 'inputs')
        with tf.variable_scope('model') as _:
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs, vgg19_path)
            self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

        print('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        checkpoint_saver = tf.train.import_meta_graph('%s.%s' % (checkpoint_path, 'meta'))
        checkpoint_saver.restore(self.session, checkpoint_path)

    def synthesize(self, images_dir, output_wav_dir):
        for path, _, filenames in os.walk(images_dir):
            for test_file in filenames:
                if str.endswith(test_file, '.png'):
                    base_file_name, _ = os.path.splitext(test_file)
                    raw_image = imread(os.path.join(path, test_file), mode='RGB')
                    processed_image = imresize(raw_image, (224, 224, 3))

                    feed_dict = {
                        self.model.inputs: [np.asarray(processed_image, dtype=np.float32)],
                    }
                    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
                    wav = audio.inv_preemphasis(wav)
                    wav = wav[:audio.find_endpoint(wav)]
                    audio_out_path = os.path.join(output_wav_dir, 'eval-{}.wav'.format(base_file_name))
                    audio.save_wav(wav, audio_out_path)
                    print('Wav - {} generated successfully!'.format(audio_out_path))
