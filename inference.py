from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    # создаем объект генератора
    generator = Generator(h).to(device)

    # просто загружаем веса модели
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # получаем список всех входных файлов
    filelist = os.listdir(a.input_wavs_dir)

    # создаем директорию для сгенерированнных файлов
    os.makedirs(a.output_dir, exist_ok=True)

    # TODO: understand
    generator.eval()
    generator.remove_weight_norm()
    # для inference, мы отключаем autograd пайторча, потому что нам больше не надо считать градиент
    with torch.no_grad():
        # итерируем по входным файлам
        for i, filname in enumerate(filelist):
            # получаем данные аудио и частоту дискретизации
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            # TODO: короче говоря обработка аудио и его прокидывание
            print('--- wav, sr,', wav, sr)
            wav = wav / MAX_WAV_VALUE
            print('--- wav / MAX_WAV_VALUE,', wav)
            wav = torch.FloatTensor(wav).to(device)
            print('--- torch.FloatTensor(wav).to(device)', wav)
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            print('--- y_g_hat', y_g_hat)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            # записываем в папку результатов и в консоль результаты работы генератора
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    # Выделяем переданные параметры
    parser = argparse.ArgumentParser()
    # директория с входными файлами
    parser.add_argument('--input_wavs_dir', default='test_files')
    # директория для результатов
    parser.add_argument('--output_dir', default='generated_files')
    # файл чекпоинтов, т.е. файл с набором весов нашей модели на разных итерациях
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    # считываем данные конфига для папки с чекпоинтами
    # TODO: почему чекпоинты разбиты по попкам
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    # преобразуем полученный ранее конфиг в питон объект для удобства работы
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # задаем ядро для RNG
    torch.manual_seed(h.seed)

    # устанавливаем используемый девайс для работы (GPU vs CPU)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # infernce это процесс ввода данных в модель машинного обучения для расчета выходных данных
    inference(a)


if __name__ == '__main__':
    main()

