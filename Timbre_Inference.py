import soundfile as sf
from IPython.display import Audio
import os
from Config import get_options
import torch
from Model import DDSP, Mamba_DDSP
import librosa as li
from preprocess import preprocess, Dataset
from core import mean_std_loudness
import numpy as np


def load_model(args):

    # ------------------ Load Model ------------------------ #
    print(f'\n\nLoading {args.model_type} Model\n\n')

    if args.model_type == "Mamba":

      model = Mamba_DDSP(hidden_size=args.hidden_size, n_harmonic=args.n_harmonic, n_bands=args.n_bands, sampling_rate=args.sampling_rate, block_size= args.block_size).to(args.device)
    else:
      model = DDSP(hidden_size=args.hidden_size, n_harmonic=args.n_harmonic, n_bands=args.n_bands, sampling_rate=args.sampling_rate, block_size= args.block_size).to(args.device)

    state = model.state_dict()

    if args.model_type == "Mamba":
        #flute_model_epoch_39999.pk
        checkpoint_path = os.path.join(args.save_dir, f'{args.model_type}', f'{args.datatype}_model_epoch_9999.pk')
    else:
        # checkpoint_path = os.path.join(args.save_dir, f'{args.model_type}_{args.datatype}_best_model.pk')
        checkpoint_path = os.path.join(args.save_dir, f'{args.datatype}_best_model2.pk')

    pretrained = torch.load(checkpoint_path, map_location=args.device)
    state.update(pretrained)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_recording(args):
    #  --------------- Load and preprocess recording -------- #

    f = args.singing_path  # Path to recording wav file

    a, sr = li.load(f, sr=args.sampling_rate, duration=10.0)

    print(a.shape)

    x, p, l = preprocess(f, args.sampling_rate, args.block_size, args.signal_length, args.oneshot)

    print(f'\n x_size - {x.shape}, p_size - {p.shape}, l_size - {l.shape}\n')

    return p, l


def normalize_pitch_loudness(p,l):
    # x = torch.from_numpy(x.astype(np.float32)).unsqueeze(-1).to(device)
    original_p = torch.from_numpy(p.astype(np.float32)).unsqueeze(-1).to(args.device)
    original_l = torch.from_numpy(l.astype(np.float32)).unsqueeze(-1).to(args.device)
    # should have size : (1, sequence_length, 1)
    print(f'p_size - {original_p.size()}, l_size - {original_l.size()}')

    # -------------- Getting same mean, variance from training -------- #
    dataset = Dataset(args.out_dir)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.BATCH,
        True,
        drop_last=True,
    )

    mean_loudness, std_loudness = mean_std_loudness(dataloader)

    l = (original_l - mean_loudness) / std_loudness

    # 🔁 Shift pitch up by 2 octaves for violin transfer
    p = original_p  # * 4.0

    return p, l


def generate_and_save(model, p, l, args):

    with torch.no_grad():
        generated_audio = model(p, l).squeeze(-1).cpu().numpy()
        print(f'g_size - {generated_audio.shape}')
        # Convert to NumPy and stack all chunks sequentially
        audio_waveform = generated_audio.reshape(-1)  # Shape: (32 * 64000,)
        print(f'audio_size - {audio_waveform.shape}')

    if args.model_type == "Mamba":
        timbre_transferred_path = args.mamba_timbre_transferred_path
    else:
        timbre_transferred_path = args.timbre_transferred_path

    sf.write(timbre_transferred_path, audio_waveform, args.sampling_rate)


if __name__ == "__main__":

    args = get_options(args=[])

    # ------------------  Loading Model  -------------------------- #

    model = load_model(args)

    # ------------- Extract Pitch, loudness from recording -------- #

    p, l = preprocess_recording(args)

    # ------------------  Normalize p,l  -------------------------- #

    norm_p, norm_l = normalize_pitch_loudness(p=p, l=l)

    # ------------------  preform and sae generation  ------------- #

    generate_and_save(model=model, p=norm_p, l=norm_l, args=args)
