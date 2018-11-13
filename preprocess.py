# coding=utf-8
import argparse
import os
import shutil
import subprocess
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from multiprocessing import cpu_count
from shutil import move

from datasets import multi_model


def __chunk_category(category, args):
    # Step 1: write all filenames to a list
    category_base_dir = os.path.join(args.base_dir, args.input, 'wavs', str(category))
    raw_files = []
    index = 0
    with open(os.path.join(category_base_dir, 'preprocess_file_list.txt'), 'w') as f:
        for path, _, filenames in os.walk(category_base_dir):
            for filename in filenames:
                if str.endswith(filename, '.wav'):
                    raw_files.append(filename)
                    index += 1
                    if args.wavs == -1 or index <= args.wavs:
                        f.write("file '" + path + '/' + filename + "'\n")

    # Step 2: concatenate everything into one massive wav file
    os.system('ffmpeg -f concat -safe 0 -i {}/preprocess_file_list.txt '
              '{}/preprocess_all_audio.wav'.format(category_base_dir, category_base_dir))

    # Delete raw audio files
    for file in raw_files:
        os.remove(os.path.join(category_base_dir, file))

    # Compute length of the resulting file
    length = float(subprocess.check_output('ffprobe -i {}/preprocess_all_audio.wav -show_entries format=duration -v '
                                           'quiet -of csv="p=0"'.format(category_base_dir), shell=True))

    # Step 3: Down sample and split the big file into chunks
    for i in range(int(length) // args.chunk_len):
        os.system('ffmpeg -ss {} -t {} -i {}/preprocess_all_audio.wav -ac 1 -ab 16k -ar 16000 {}/{}p{}.wav'
                  .format(i * args.chunk_len, args.chunk_len, category_base_dir, category_base_dir, category, i))

    # # Step 4: clean up temp files
    os.remove(os.path.join(category_base_dir, 'preprocess_all_audio.wav'))
    os.remove(os.path.join(category_base_dir, 'preprocess_file_list.txt'))


def __chunk_audio_samples(args):
    for i in range(args.categories):
        __chunk_category(i, args)


def __preprocess_multi_model(args):
    """
    Pre-processes the audio-image pair in parallel!
    :param args: configs required for preprocessing
    """
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = multi_model.build(args)
    __write_metadata(metadata, out_dir)


def __write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(metadata)

    print('Raw data preprocessed and dataset generated!')


def __fetch_pre_trained_model(args):
    out_path = os.path.join(args.base_dir, args.output, 'vgg19')
    os.makedirs(out_path, exist_ok=True)
    move(os.path.join(args.base_dir, args.vgg19_model_path), out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/Songbird'))
    parser.add_argument('--output', default='training')
    parser.add_argument('--input', default='raw_data')
    parser.add_argument('--categories', type=int, default=6)
    parser.add_argument('--chunk_len', type=int, default=5)
    parser.add_argument('--imgs', help='Number of images to add to the dataset per category', type=int, default=-1)
    parser.add_argument('--wavs', help='Number of Audio files to add to the dataset per category', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--clean_up', type=bool, default=True)
    parser.add_argument('--vgg19_model_path', default='vgg19.npy', help='Relative path to the pre-trained VGG-19 model')
    args = parser.parse_args()
    __chunk_audio_samples(args)
    __preprocess_multi_model(args)
    __fetch_pre_trained_model(args)
    if args.clean_up:
        shutil.rmtree(os.path.join(args.base_dir, args.input))


if __name__ == "__main__":
    main()
