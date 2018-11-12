import argparse
import datetime
import os

from synthesizer import Synthesizer


def __get_output_wav_path(args):
    output_dir = os.path.join(args.base_dir, 'out')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return os.path.join(output_dir, 'eval-{}.wav'.format(timestamp))


def __run_eval(args):
    synth = Synthesizer()
    synth.load(args.checkpoint, os.path.join(args.base_dir, args.vgg19_path))
    wav_path = __get_output_wav_path(args)
    print('Synthesizing: %s' % wav_path)
    with open(wav_path, 'wb') as f:
        f.write(synth.synthesize(args.image_path))
    print('Wav - {} generated successfully!'.format(wav_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--image_path', required=True, help='Path to the test image')
    parser.add_argument('--base_dir', default=os.path.expanduser('~/Songbird'))
    parser.add_argument('--vgg19_path', default='training/vgg19/vgg19.npy')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    __run_eval(args)


if __name__ == '__main__':
    main()
