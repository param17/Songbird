# coding=utf-8
import argparse
import os
import subprocess
import shutil
import itertools


def __input_wav_dir(args, category):
    return os.path.join(args.base_dir, args.input, 'wavs', str(category))


def __chunk_category(category, args):
    # Step 1: write all filenames to a list
    category_base_dir = __input_wav_dir(args, category)
    raw_files = []
    with open(os.path.join(category_base_dir, 'preprocess_file_list.txt'), 'w') as f:
        for path, _, filenames in os.walk(category_base_dir):
            for filename in filenames:
                if str.endswith(filename, '.wav'):
                    raw_files.append(filename)
                    f.write("file '" + path + '/' + filename + "'\n")

    # Step 2: concatenate everything into one massive wav file
    category_output_dir = os.path.join(category_base_dir, 'preprocessed')
    os.makedirs(category_output_dir)
    os.system('ffmpeg -f concat -safe 0 -i {}/preprocess_file_list.txt '
              '{}/preprocess_all_audio.wav'.format(category_base_dir, category_output_dir))

    # Delete raw audio files
    for file in raw_files:
        os.remove(os.path.join(category_base_dir, file))

    # Compute length of the resulting file
    length = float(subprocess.check_output('ffprobe -i {}/preprocess_all_audio.wav -show_entries format=duration -v '
                                           'quiet -of csv="p=0"'.format(category_output_dir), shell=True))

    # Step 3: Down sample and split the big file into chunks
    for i in range(int(length) // args.chunk_len):
        os.system('ffmpeg -ss {} -t {} -i {}/preprocess_all_audio.wav -ac 1 -ab 16k -ar 16000 {}/{}p{}.wav'
                  .format(i * args.chunk_len, args.chunk_len, category_output_dir, category_output_dir, category, i))

    # # Step 4: clean up temp files
    os.remove(os.path.join(category_output_dir, 'preprocess_all_audio.wav'))
    os.remove(os.path.join(category_base_dir, 'preprocess_file_list.txt'))


def chunk_audio_samples(args):
    for i in range(args.categories):
        __chunk_category(i, args)


def __move_images(category, args, img_output_base_dir):
    category_input_base_dir = os.path.join(args.base_dir, args.input, 'imgs', str(category))
    index = 0
    for path, _, filenames in os.walk(category_input_base_dir):
        for file in filenames:
            output_filename = '{}p{}.png'.format(category, index)
            shutil.move(os.path.join(os.path.join(category_input_base_dir, file)),
                        os.path.join(img_output_base_dir, output_filename))
            index += 1
    return index


def __move_chunked_audio(category, args, wav_output_base_dir):
    category_input_base_dir = os.path.join(__input_wav_dir(args, category), 'preprocessed')
    index = 0
    for path, _, filenames in os.walk(category_input_base_dir):
        for file in filenames:
            output_filename = '{}p{}.wav'.format(category, index)
            shutil.move(os.path.join(os.path.join(category_input_base_dir, file)),
                        os.path.join(wav_output_base_dir, output_filename))
            index += 1
    return index


def __write_metadata(base_output_dir, category, image_count, audio_count):
    with open(os.path.join(base_output_dir, 'metadata.txt'), 'a') as f:
        for image_index, audio_index in itertools.product(range(image_count), range(audio_count)):
            image_file_name = '{}p{}.png'.format(category, image_index)
            audio_file_name = '{}p{}.wav'.format(category, audio_index)
            f.write('{}|{}\n'.format(audio_file_name, image_file_name))


def create_dataset(args):
    base_output_dir = os.path.join(args.base_dir, args.output)

    img_output_base_dir = os.path.join(base_output_dir, 'imgs')
    wav_output_base_dir = os.path.join(base_output_dir, 'wavs')
    os.makedirs(img_output_base_dir, exist_ok=True)
    os.makedirs(wav_output_base_dir, exist_ok=True)

    for category in range(args.categories):
        image_count = __move_images(category, args, img_output_base_dir)
        audio_count = __move_chunked_audio(category, args, wav_output_base_dir)
        __write_metadata(base_output_dir, category, image_count, audio_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
    parser.add_argument('--output', default='MultiModalDataset-1.0')
    parser.add_argument('--input', default='raw_data')
    parser.add_argument('--categories', type=int, default=6)
    parser.add_argument('--chunk_len', type=int, default=5)
    args = parser.parse_args()
    chunk_audio_samples(args)
    create_dataset(args)


if __name__ == "__main__":
    main()
