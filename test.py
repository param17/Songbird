# coding=utf-8
import argparse
import os
import traceback

import tensorflow as tf

from datasets.datafeeder import DataFeeder
from hparams import hparams
from models import create_model
from util import audio, infolog

log = infolog.log


def test(log_dir, args):
    input_path = os.path.join(args.base_dir, args.input)

    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as _:
        feeder = DataFeeder(coord, input_path, hparams)

    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('model') as _:
        model = create_model(args.model, hparams)
        model.initialize(feeder.inputs, args.vgg19_pretrained_model, feeder.mel_targets, feeder.linear_targets, feeder.input_names)
        model.add_loss()
        model.add_optimizer(global_step)

    # Test!
    with tf.Session() as sess:
        try:
            sess.run(tf.global_variables_initializer())

            # Restore from a checkpoint if the user requested it.
            checkpoint_path = os.path.join(log_dir, 'model.ckpt')
            restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
            checkpoint_saver = tf.train.import_meta_graph('%s.%s' % (restore_path, 'meta'))
            checkpoint_saver.restore(sess, restore_path)
            log('Resuming from checkpoint: %s' % restore_path)
            feeder.start_in_session(sess)

            test_images = list(set(sess.run(model.input_names)))
            for img_name in test_images:
                index = test_images.index(img_name)
                sess.run([global_step, model.loss, model.optimize])
                spectrogram = sess.run(model.linear_outputs[index])
                waveform = audio.inv_spectrogram(spectrogram.T)
                wav_name = 'eval-%s-audio.wav' % img_name.decode()
                log('Saving audio %s' % wav_name)
                audio_path = os.path.join(args.base_dir, args.output_dir, wav_name)
                audio.save_wav(waveform, audio_path)

            coord.request_stop()

        except Exception as e:
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/Songbird'))
    parser.add_argument('--input', default='training/train.txt')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--vgg19_pretrained_model', default='training/vgg19/vgg19.npy')
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--restore_step', type=int, default=36000, help='Global step to restore from checkpoint.')
    parser.add_argument('--output_dir', default='out')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    log_dir = os.path.join(args.base_dir, 'logs-%s' % args.model)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, args.output_dir), exist_ok=True)
    infolog.init(os.path.join(log_dir, 'test.log'), args.model, None)
    hparams.parse(args.hparams)
    test(log_dir, args)


if __name__ == '__main__':
    main()
