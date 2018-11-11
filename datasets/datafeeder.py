import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from util.infolog import log

_batches_per_group = 32
_pad = 0


class DataFeeder(threading.Thread):
    """Feeds batches of data into a queue on a background thread."""

    def __init__(self, coordinator, metadata_filename, hparams):
        super(DataFeeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._offset = 0

        # Load metadata:
        self._datadir = os.path.dirname(metadata_filename)
        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            hours = sum((int(x[2]) for x in self._metadata)) * hparams.frame_shift_ms / (3600 * 1000)
            log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))

        # Create placeholders for inputs and targets. Don't specify batch size because we want to
        # be able to feed different sized batches at eval time.
        self._placeholders = [
            tf.placeholder(tf.float32, [None, self._hparams.image_dim, self._hparams.image_dim, 3], 'inputs'),
            tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
            tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets')
        ]

        # Create queue for buffering data:
        queue = tf.FIFOQueue(8, [tf.float32, tf.float32, tf.float32], name='input_queue')
        self._enqueue_op = queue.enqueue(self._placeholders)
        self.inputs, self.mel_targets, self.linear_targets = queue.dequeue()
        self.inputs.set_shape(self._placeholders[0].shape)
        self.mel_targets.set_shape(self._placeholders[1].shape)
        self.linear_targets.set_shape(self._placeholders[2].shape)

    def start_in_session(self, session):
        self._session = session
        self.start()

    def run(self):
        try:
            while not self._coord.should_stop():
                self._enqueue_next_group()
        except Exception as e:
            traceback.print_exc()
            self._coord.request_stop(e)

    def _enqueue_next_group(self):
        start = time.time()

        # Read a group of examples:
        n = self._hparams.batch_size
        r = self._hparams.outputs_per_step
        examples = [self._get_next_example() for _ in range(n * _batches_per_group)]

        # Bucket examples based on similar output sequence length for efficiency:
        examples.sort(key=lambda x: x[-1])
        batches = [examples[i:i + n] for i in range(0, len(examples), n)]
        random.shuffle(batches)

        log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
        for batch in batches:
            feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
            self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        """Loads a single example (input, mel_target, linear_target, cost) from disk"""
        if self._offset >= len(self._metadata):
            self._offset = 0
            random.shuffle(self._metadata)
        meta = self._metadata[self._offset]
        self._offset += 1

        image = np.load(os.path.join(self._datadir, meta[3]))
        assert image.shape == (self._hparams.image_dim, self._hparams.image_dim, 3)
        input_data = image.astype('float32')
        linear_target = np.load(os.path.join(self._datadir, meta[0]))
        mel_target = np.load(os.path.join(self._datadir, meta[1]))
        return input_data, mel_target, linear_target, len(linear_target)


def _prepare_batch(batch, outputs_per_step):
    random.shuffle(batch)
    inputs = np.asarray([x[0] for x in batch])
    mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
    linear_targets = _prepare_targets([x[2] for x in batch], outputs_per_step)
    return inputs, mel_targets, linear_targets


def _prepare_targets(targets, alignment):
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder
