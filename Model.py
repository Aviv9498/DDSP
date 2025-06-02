import torch
import torch.nn as nn
from core import mlp, gru, scale_function, remove_above_nyquist, upsample
from core import harmonic_synth, amp_to_impulse_response, fft_convolve
from core import resample
from core import mamba_block
import math


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)  # (sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness):

        # Step 1: Extract Features from Pitch & Loudness
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        # Step 2: Temporal Modeling with GRU
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)

        # Step 3: Additional Processing via MLP
        print(f'hidden size: {hidden.size()}')
        hidden = self.out_mlp(hidden)

        # Step 4: Harmonic Synthesis
        param = scale_function(self.proj_matrices[0](hidden))
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        # Step 5: Remove Frequencies Above Nyquist Limit
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )

        # Step 6: Normalize Harmonics & Apply Total Amplitude
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        # Step 7: Generate Harmonic Signal
        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)
        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # Step 8: Noise Synthesis
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        # Step 9: Sum Harmonic + Noise Components
        # signal = harmonic
        signal = harmonic + noise

        print(f'signal input for reverb size- {signal.size()}')
        # Step 10: Apply Reverb
        signal = self.reverb(signal)

        return signal


class Mamba_DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.mamba = mamba_block(d_model=2 * hidden_size, d_state=64, d_conv=4, expand=4)
        self.out_mlp1 = nn.Linear(hidden_size * 2 + 2, hidden_size + 2)

        self.out_mlp2 = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)  # (sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness):

        # Step 1: Extract Features from Pitch & Loudness
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        # Step 2: Temporal Modeling with Mamba
        # print(f'\n hidden size: {hidden.size()}\n')
        hidden = self.mamba(hidden)
        hidden = torch.cat([hidden, pitch, loudness], -1)

        # print(f'\n hidden size after mamba: {hidden.size()}\n')
        # Step 3: Additional Processing via MLP
        hidden = self.out_mlp1(hidden)

        # print(f'\n hidden size after mlp1: {hidden.size()}\n')


        hidden = self.out_mlp2(hidden)


        # Step 4: Harmonic Synthesis
        param = scale_function(self.proj_matrices[0](hidden))
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        # Step 5: Remove Frequencies Above Nyquist Limit
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )

        # Step 6: Normalize Harmonics & Apply Total Amplitude
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        # Step 7: Generate Harmonic Signal
        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)
        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # Step 8: Noise Synthesis
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        # Step 9: Sum Harmonic + Noise Components
        # signal = harmonic
        signal = harmonic + noise

        # print(f'signal input for reverb size- {signal.size()}')
        # Step 10: Apply Reverb
        signal = self.reverb(signal)

        return signal
