import datetime
import time
import torch as th
import numpy as np
import implementation.data_processing.DataLoader as dl
import argparse
import yaml
import os
import pickle
import timeit

from torch.backends import cudnn

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="store", type=str, default="configs/2.conf",
                         help="default configuration for the Network")
    parser.add_argument("--generator_file", action="store", type=str,
                         default="training_runs/2/saved_models/GAN_GEN_3.pth",
                         help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--encoder_file", action="store", type=str, default="training_runs/2/saved_models/Encoder_3.pth",
                        help="pretrained Encoder file (compatible with my code)")
    parser.add_argument("--ca_file", action="store", type=str, default="training_runs/2/saved_models/Condition_Augmentor_3.pth",
                        help="pretrained Conditioning Augmentor file (compatible with my code)")
    args = parser.parse_args()

    return args


def get_config(conf_file):
    """
    parse and load the provided configuration
    :param conf_file: configuration file
    :return: conf => parsed configuration
    """
    from easydict import EasyDict as edict

    with open(conf_file, "r") as file_descriptor:
        data = yaml.load(file_descriptor)

    # convert the data into an easyDictionary
    return edict(data)


def create_grid(samples, scale_factor, img_file, real_imgs=False):
    """
    utility function to create a grid of GAN samples
    :param samples: generated samples for storing
    :param scale_factor: factor for upscaling the image
    :param img_file: name of file to write
    :param real_imgs: turn off the scaling of images
    :return: None (saves a file)
    """
    from torchvision.utils import save_image
    from torch.nn.functional import interpolate

    samples = th.clamp((samples / 2) + 0.5, min=0, max=1)

    # upsample the image
    if not real_imgs and scale_factor > 1:
        samples = interpolate(samples,
                              scale_factor=scale_factor)

    # save the images:
    save_image(samples, img_file, nrow=int(np.sqrt(len(samples))))


def create_descriptions_file(file, captions, dataset):
    """
    utility function to create a file for storing the captions
    :param file: file for storing the captions
    :param captions: encoded_captions or raw captions
    :param dataset: the dataset object for transforming captions
    :return: None (saves a file)
    """
    from functools import reduce

    # transform the captions to text:
    if isinstance(captions, th.Tensor):
        captions = list(map(lambda x: dataset.get_english_caption(x.cpu()),
                            [captions[i] for i in range(captions.shape[0])]))

        with open(file, "w") as filler:
            for caption in captions:
                filler.write(reduce(lambda x, y: x + " " + y, caption, ""))
                filler.write("\n\n")
    else:
        with open(file, "w") as filler:
            for caption in captions:
                filler.write(caption)
                filler.write("\n\n")


def main(args):
    from implementation.networks.TextEncoder import Encoder
    from implementation.networks.ConditionAugmentation import ConditionAugmentor

    from pro_gan_pytorch.PRO_GAN import ConditionalProGAN
    config = get_config(args.config)
    if args.generator_file is not None:
        print("Loading generator from:", args.generator_file)
        c_pro_gan = ConditionalProGAN(
            embedding_size=config.hidden_size,
            depth=config.depth,
            latent_size=config.latent_size,
            compressed_latent_size=config.compressed_latent_size,
            learning_rate=config.learning_rate,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
            eps=config.eps,
            drift=config.drift,
            n_critic=config.n_critic,
            use_eql=config.use_eql,
            loss=config.loss_function,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay,
            device=device
        )
    c_pro_gan.gen.load_state_dict(th.load(args.generator_file))
    dataset = dl.RawTextFace2TextDataset(
        annots_file=config.annotations_file,
        img_dir=config.images_dir,
        img_transform=dl.get_transform(config.img_dims)
    )
    from implementation.networks.TextEncoder import PretrainedEncoder
    # create a new session object for the pretrained encoder:
    text_encoder = PretrainedEncoder(
        model_file=config.pretrained_encoder_file,
        embedding_file=config.pretrained_embedding_file,
        device=device
    )
    text_encoder.load_state_dict(th.load(args.encoder_file))
    condition_augmenter = ConditionAugmentor(
        input_size=config.hidden_size,
        latent_size=config.ca_out_size,
        use_eql=config.use_eql,
        device=device
    )
    condition_augmenter.load_state_dict(th.load(args.ca_file))
    sample_dir = "training_runs/2/generated_samples/"

    fixed_save_dir = os.path.join(sample_dir, "__Real_Info")
    temp_data = dl.get_data_loader(dataset, 1, num_workers=3)
    tempp = iter(temp_data)
    #print(next(tempp))
    fixed_captions, fixed_real_images = next(tempp)
    fixed_captions = ("black woman, smiling, long nose",)
    print(type(fixed_captions))
    create_descriptions_file(os.path.join(fixed_save_dir, "sampleInputforTest.txt"),
                             fixed_captions,
                             dataset)

    fixed_embeddings = text_encoder(fixed_captions)
    fixed_embeddings = th.from_numpy(fixed_embeddings).to(device)
    #print(fixed_embeddings)
    fixed_c_not_hats, _, _ = condition_augmenter(fixed_embeddings)

    fixed_noise = th.randn(len(fixed_captions),
                           c_pro_gan.latent_size - fixed_c_not_hats.shape[-1]).to(device)

    fixed_gan_input = th.cat((fixed_c_not_hats, fixed_noise), dim=-1)
    current_depth = 3
    alpha = 1
    sample_dir = config.sample_dir
    gen_img_file = os.path.join(sample_dir, "Final"+".png")
    create_grid(
        samples=c_pro_gan.gen(
            fixed_gan_input,
            current_depth,
            alpha
        ),
        scale_factor=int(np.power(2, c_pro_gan.depth - current_depth - 1)),
        img_file=gen_img_file,
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())