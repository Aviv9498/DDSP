import wandb
import torch
import yaml
from Model import DDSP, Mamba_DDSP
from os import path
from preprocess import Dataset
from tqdm import tqdm
from core import multiscale_fft, safe_log, mean_std_loudness
from utils import get_scheduler
import numpy as np
from datetime import datetime
from Config import get_options
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(args):

    # ------------ WANDB Tracking----------------------#

    if args.WANDB_TRACKING:
        wandb.login(key="3ec39d34b7882297a057fdc2126cd037352175a4")

        # Generate a unique timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize the WandB run
        wandb.init(
            project="DDSP",
            name=f"{args.model_type}_{args.datatype}_{timestamp}",  # Add timestamp to the run name
            config={

                "epochs": args.STEPS,
                "batch_size": args.BATCH,
                "lr": args.START_LR,

            }
        )

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model_type == "Mamba":
        model = Mamba_DDSP(hidden_size=args.hidden_size, n_harmonic=args.n_harmonic, n_bands=args.n_bands,
                           sampling_rate=args.sampling_rate, block_size=args.block_size).to(device)
    else:
        model = DDSP(hidden_size=args.hidden_size, n_harmonic=args.n_harmonic, n_bands=args.n_bands,
                     sampling_rate=args.sampling_rate, block_size=args.block_size).to(device)

    # -------------------------------- Load pre-trained ------------------------------------- #
    if args.load_pretrained:
        state = model.state_dict()
        checkpoint_path = os.path.join(args.save_dir, f'{args.datatype}_best_model.pk')
        pretrained = torch.load(checkpoint_path, map_location=device)
        state.update(pretrained)
        model.load_state_dict(state)
    # ---------------------------------------------------------------------------------------- #

    # --------------------- Create Dataset -------------------------------- #
    dataset = Dataset(args.out_dir)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.BATCH,
        True,
        drop_last=True,
    )

    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    args.mean_loudness = mean_loudness
    args.std_loudness = std_loudness

    opt = torch.optim.Adam(model.parameters(), lr=args.START_LR, weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience,
                                  verbose=True)

    best_loss = float("inf")
    mean_loss = 0
    epochs = int(np.ceil(args.STEPS / len(dataloader)))

    # ----------------------- Start Training ---------------- #

    train_losses = []

    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for s, p, l in dataloader:
            s = s.to(device)

            # print(f'\n s_size- {s.size()}, p_size - {p.size()}, l_size - {l.size()}\n')

            p = p.unsqueeze(-1).to(device)
            l = l.unsqueeze(-1).to(device)

            # print(f'\np_size - {p.size()}, l_size - {l.size()}\n')

            l = (l - mean_loudness) / std_loudness

            y = model(p, l).squeeze(-1)

            # print(f'\n y_size- {y.size()}\n')

            ori_stft = multiscale_fft(
                s,
                args.scales,
                args.overlap,
            )
            rec_stft = multiscale_fft(
                y,
                args.scales,
                args.overlap,
            )

            loss = 0
            for s_x, s_y in zip(ori_stft, rec_stft):
                lin_loss = (s_x - s_y).abs().mean()
                log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
                loss = loss + lin_loss + log_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(dataloader))

        if args.use_scheduler:
            scheduler.step(train_losses[-1])  # Step the scheduler with the validation loss

        if args.WANDB_TRACKING:
            wandb.log({
                "loss": train_losses[-1],
                "lr": opt.param_groups[0]['lr'],
                "reverb_decay": model.reverb.decay.item(),
                "reverb_wet": model.reverb.wet.item()
            }, step=e)

        # Printing to prompt
        if (e + 1) % args.print_every == 0:
            print(f'Epoch {e + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}')

        # Check if Model improved
        if mean_loss < best_loss:
            best_loss = mean_loss
            args.save_dir = os.path.join(args.save_dir, f'{args.model_type}')
            os.makedirs(args.save_dir, exist_ok=True)  # ✅ create the directory if it doesn't exist
            checkpoint_path = os.path.join(args.save_dir, f'{args.datatype}_best_model.pk')
            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

        # Printing to prompt
        if (e + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'{args.datatype}_model_epoch_{e}.pk')
            torch.save(
                model.state_dict(),
                checkpoint_path,
            )
            print(f'\n\n Saved Model at Epoch --- {e}\n\n')

    if args.WANDB_TRACKING:
        wandb.finish()


if __name__ == "__main__":
    args = get_options(args=[])
    wandb.finish()
    train(args=args)






