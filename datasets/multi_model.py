import os
import itertools
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import trange

import numpy as np
from scipy.misc import imread, imresize

from util import audio


def __load_and_save_images(category, config):
    category_input_base_dir = os.path.join(config.base_dir, config.input, 'imgs', str(category))
    index = 0
    image_limit = config.imgs
    for path, _, filenames in os.walk(category_input_base_dir):
        for file in filenames:
            if image_limit != -1 and index >= image_limit:
                break

            image_path = os.path.join(category_input_base_dir, file)
            raw_image = imread(image_path, mode='RGB')
            # Pre-process image
            preprocessed_image = imresize(raw_image, (224, 224, 3))
            # Write image to disk
            preprocessed_image_filename = '{}bird{}.npy'.format(category, index)
            np.save(os.path.join(config.base_dir, config.output, preprocessed_image_filename), preprocessed_image,
                    allow_pickle=False)
            index += 1
    return index


def __preprocess_audio(category, config):
    category_input_base_dir = os.path.join(config.base_dir, config.input, 'wavs', str(category))
    out_dir = os.path.join(config.base_dir, config.output)
    index = 0
    audio_limit = config.wavs
    for path, _, filenames in os.walk(category_input_base_dir):
        for file in filenames:
            if audio_limit != -1 and index >= audio_limit:
                break

            __generate_spectrograms(os.path.join(category_input_base_dir, file), category, index, out_dir)

            index += 1
    return index


def __generate_spectrograms(file_path, category, index, out_dir):
    wav = audio.load_wav(file_path)
    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    # Write the spectrograms to disk:
    spectrogram_filename = '{}spec{}.npy'.format(category, index)
    mel_filename = '{}mel{}.npy'.format(category, index)
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)


def __generate_metadata(category, image_count, audio_count):
    meta_data = []
    for image_index, audio_index in itertools.product(range(image_count), range(audio_count)):
        image_file_name = '{}bird{}.npy'.format(category, image_index)
        spec_file_name = '{}spec{}.npy'.format(category, audio_index)
        mel_file_name = '{}mel{}.npy'.format(category, audio_index)
        meta_data.append('{}|{}|{}\n'.format(spec_file_name, mel_file_name, image_file_name))
    return meta_data


def __preprocess_in_parallel(category, config):
    image_count = __load_and_save_images(category, config)
    audio_count = __preprocess_audio(category, config)
    return __generate_metadata(category, image_count, audio_count)


def build(config):
    executor = ProcessPoolExecutor(max_workers=config.num_workers)
    meta_data = []
    for category in trange(config.categories, desc='# Category progress'):
        meta_data += executor.submit(partial(__preprocess_in_parallel, category, config)).result()
    return meta_data
