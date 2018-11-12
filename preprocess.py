# coding=utf-8
import argparse
import os
from shutil import move
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import multi_model
from hparams import hparams


def preprocess_multi_model(args):
    """
    Pre-processes the audio-image pair in parallel!
    :param args: configs required for preprocessing
    """
    in_dir = os.path.join(args.base_dir, 'MultiModalDataset-1.0')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = multi_model.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    frames = 0
    max_output = 0
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
            frames += m[2]
            max_output = max(max_output, m[2])

    hours = frames * hparams.frame_shift_ms / (3600 * 1000)
    print('Wrote %d chirps, %d frames (%.2f hours)' % (len(metadata), frames, hours))
    print('Max output length: %d' % max_output)


def fetch_pre_trained_model(args):
    out_path = os.path.join(args.base_dir, args.output, 'vgg19')
    os.makedirs(out_path, exist_ok=True)
    move(os.path.join(args.base_dir, args.vgg19_model_path), out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/Songbird'))
    parser.add_argument('--output', default='training')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--vgg19_model_path', default='vgg19.npy', help='Relative path to the pre-trained VGG-19 model')
    args = parser.parse_args()
    preprocess_multi_model(args)
    fetch_pre_trained_model(args)


if __name__ == "__main__":
    main()
