import yaml
import pathlib
import librosa as li
from core import extract_loudness, extract_pitch
# from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from Config import get_options
from scipy.io import wavfile


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    # print(f'\n{f}\n')
    x, sr = li.load(f, sr=sampling_rate)  #, duration=30.0)

    # x = x[:64000]
    print(f'\nShape after Loading{x.shape}\n')
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    print(f'\nx Shape before pitch {x.shape}\n')
    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l

def main():

    args = get_options(args=[])

    # class args(Config):
        # CONFIG = "config.yaml"

    # args.parse_args()
    # with open(args.CONFIG, "r") as config:
        # config = yaml.safe_load(config)

    files = get_files(data_location=args.data_location, extension=args.extension)
    # files = files[:100]
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l = preprocess(f, args.sampling_rate, args.block_size, args.signal_length, args.oneshot )
        print(f'\n x_size - {x.shape}, p_size - {p.shape}, l_size - {l.shape}\n')
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    out_dir = args.out_dir
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)


if __name__ == "__main__":
    main()