"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import torch as th
import torch.distributed as dist
import torchvision as tv
from torchvision.utils import draw_bounding_boxes

from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from lostGAN.data import *


def main():
    args = create_argparser().parse_args()
    print(f"cuda available: {th.cuda.is_available()}")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    dataset = get_dataset(args.image_size)
    data = th.utils.data.Dataloader(
        dataset, batch_size=args.batch_size,
        drop_last=True, shuffle=True, num_workers=0,
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    layout_path = os.path.join(args.results_path, 'layout')
    os.makedirs(layout_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    evaluation_path = os.path.join(args.results_path, 'evaluation')
    os.makedirs(evaluation_path, exist_ok=True)

    logger.log("sampling...")
    all_samples = []
    for i, (image, cond) in enumerate(data):
        #image = ((batch + 1.0) / 2.0).cuda()
        model_kwargs = cond
        bbox = model_kwargs['boxes']

        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, image.shape[2], image.shape[3]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
        sample = (sample + 1) / 2.0

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(sample.shape[0]):
            image_boxes = bbox_preprocessing(bbox[j]).to(th.int32)
            image_final = image_preprocessing(image[j]).to(th.uint8)
            sample_final = sample_preprocessing(sample[j]).to(th.uint8)
            layout_image = draw_bounding_boxes(image_final, image_boxes, width=2)
            image_sample_layout = [image_final.to(dist_util.dev()), sample_final.to(dist_util.dev())]
            grid = make_grid(image_sample_layout)
            tv.utils.save_image(image_final.float(), os.path.join(image_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(sample_final.float(), os.path.join(sample_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(layout_image.float(), os.path.join(layout_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(grid.float(), os.path.join(evaluation_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

        if len(all_samples) * args.batch_size > args.num_samples:
            break

    dist.barrier()
    logger.log("sampling complete")


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['objs'] = data['objs'].long()
    data['boxes'] = data['boxes'].long()

    if self.drop_rate > 0.0:
        mask_objs?????????????????????????????????????????????????????????????????????????????????????????

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
