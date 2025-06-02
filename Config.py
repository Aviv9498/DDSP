import argparse
import torch
import os


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters for training")

    # DATA
    parser.add_argument('--data_location', default='/content/gdrive/Shareddrives/GenAI_Audio/DDSP/solo_violin_data', help='dataset path')
    parser.add_argument('--extension', default="mp3", help='file format') #"wav"

    # preprocess
    parser.add_argument('--sampling_rate', default=16000, help='sample rate')  #1600
    parser.add_argument('--signal_length', default=64000, help='signal_length')
    parser.add_argument('--block_size', default=160, help='train model name') # must be a power of 2 if using realtime
    parser.add_argument('--oneshot', default=False, help='crop every audio file to exactly signal length')
    parser.add_argument('--out_dir', default=r'/home/beaviv/DDSP/preprocessed/flute', help='path to save processed data')


    # Model
    parser.add_argument('--datatype', default="flute", help='flute/violin')
    parser.add_argument('--model_type', default="Mamba", help='')  # Mamba\GRU

    parser.add_argument('--hidden_size', default=512, help='model hidden size')
    parser.add_argument('--n_harmonic', default=100, help='')
    parser.add_argument('--n_bands', default=65, help='')
    # parser.add_argument('--sampling_rate', default=16000, help='')
    # parser.add_argument('--block_size', default=160, help='') Reverb_length
    parser.add_argument('--Reverb_length', default=16000, help='here its same as sampling rate but in the og implement its 48000, want to try')


    # Training
    parser.add_argument('--scales', default=[4096, 2048, 1024, 512, 256, 128], help='')
    parser.add_argument('--overlap', default=.75, help='')

    parser.add_argument('--STEPS', default=1000000, help='epochs')
    parser.add_argument('--BATCH', default=32, help='batch_size')  # 64
    parser.add_argument('--START_LR', default=1e-3, help='learning_rate')
    parser.add_argument('--STOP_LR', default=1e-5, help='')
    parser.add_argument('--DECAY_OVER', default=400000, help='')
    parser.add_argument('--ROOT', default="/content/gdrive/Shareddrives/GenAI_Audio/DDSP/runs", help='') # root for folder path, NAME = "debug"
    parser.add_argument('--NAME', default="debug", help='') # root for folder path, NAME = "debug"
    parser.add_argument('--load_pretrained', default=False, help='weather to load saved model or not')



    parser.add_argument('--WANDB_TRACKING', type=bool, default=True, help="weather to log training to WANDB")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="l2 regularization parameter")
    parser.add_argument('--use_scheduler', type=bool, default=False, help="weather to use scheduler")
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help="in how much to divide the lr")
    parser.add_argument('--scheduler_patience', type=int, default=100,
                        help="amount of episodes with no improvement to wait for lr reduction")
    parser.add_argument('--print_every', type=int, default=10, help="once in how many episodes to wait before printing")

    parser.add_argument('--save_every', type=int, default=500, help="once in how many episodes to save model")

    # performance
    parser.add_argument('--N_RUN', default=10, help="")

    # export
    parser.add_argument('--RUN', default=None, help="")
    parser.add_argument('--DATA', default=False, help="")
    parser.add_argument('--OUT_DIR', default="/content/gdrive/Shareddrives/GenAI_Audio/DDSP/export", help="")
    parser.add_argument('--REALTIME', default=False, help="")

    # Timbre Inference
    parser.add_argument('--singing_path', default=r'/home/beaviv/DDSP/Inference_Mamba/Butterfly_lovers_Violin.wav', help="")  # timbre_transferred.wav
    parser.add_argument('--timbre_transferred_path', default="/home/beaviv/DDSP/Inference/timbre_transferred_violin_to_flute_real_violin.wav", help="")  # timbre_transferred.wav  reverbtry
    parser.add_argument('--mamba_timbre_transferred_path', default=r'/home/beaviv/DDSP/Inference_Mamba/timbre_transferred_violin_to_flute_Butterfly_9999.wav', help="")  # timbre_transferred.wav  reverbtry


    # Run Test \ Run Train

    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    parser.add_argument('--save_dir', type=str,
                        default=r'/home/beaviv/DDSP/trained_models',
                        help="folder path for trained models")
    parser.add_argument('--trained_model_path', type=str,
                        default=r'/content/Shareddrives/GenAI_Audio/collab/snd_traffic_prediction_new_version/trained_models',
                        help="folder path for trained models")
    parser.add_argument('--training', type=bool, default=True, help="whether to train model")
    parser.add_argument('--testing', type=bool, default=False, help="whether to test model")

    opts = parser.parse_args(args)
    opts = parser.parse_args(args)

    return opts