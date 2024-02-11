import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

import torchvision as tv
from torchvision.utils import make_grid, draw_bounding_boxes
from pathlib import Path

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        num_classes,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        drop_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        clip_denoised=False,
        results_path='./OUTPUT',
        use_ddim=False
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.clip_denoised = clip_denoised
        self.results_path = results_path
        self.use_ddim = use_ddim

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    th.load(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

            if self.opt.param_groups[0]['lr'] != self.lr:
                self.opt.param_groups[0]['lr'] = self.lr

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(iter(self.data)) # make the dataloader iterable
            #cond = self.preprocess_input(cond) # No preprocessing for COCO dataset here
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0 and self.step != 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step != 0: 
                self.save()
                self.save_image()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()
    
    def save_images(self):
        self.model.eval()
        image_path = os.path.join(self.results_path, 'images')
        os.makedirs(image_path, exist_ok=True)
        sample_path = os.path.join(self.results_path, 'sample')
        os.makedirs(sample_path, exist_ok=True)
        layout_path = os.path.join(self.results_path, 'layout')
        os.makedirs(layout_path, exist_ok=True)
        evaluate_path = os.path.join(self.results_path, 'evaluate')
        os.makedirs(evaluate_path, exist_ok=True)

        all_samples = []
        image, cond = next(iter(self.data))
        model_kwargs = cond
        bbox = model_kwargs['boxes']

        sample_fn = (
            self.model,
            (self.batch_size, 3, image.shape[2], image.shape[3]),
            clip_denoised=self.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
        sample = (sample + 1) / 2.0 # turn the range from (-1, 1) to (0, 1)

        gathered_samples = [th.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy for sample in gathered_samples])

        for j in range(sample.shape[0]):
            image_boxes = bbox_preprocessing(bbox[j]).to(th.int32)
            image_final = image_preprocessing(image[j]).to(th.uint8)
            sample_final = sample_preprocessing(sample[j]).to(th.uint8)
            layout_image = draw_bounding_boxes(image_final, image_boxes, width=2)
            image_sample_layout = [image_final.to(dist_util.dev()), sample_final.to(dist_util.dev(), layout_image.to(dist_util.dev()))]
            grid = make_grid(image_sample_layout)
            tv.utils.save_image(image_final.float(), os.path.join(image_path, f"image{(self.step+self.resume_step):06d}_{j}.png"), normalize=True)
            tv.utils.save_image(sample_final.float(), os.path.join(sample_path, f"sample{(self.step+self.resume_step):06d}_{j}.png"), normalize=True)
            tv.utils.save_image(layout_image.float(), os.path.join(layout_path, f"layout{(self.step+self.resume_step):06d}_{j}.png"), normalize=True)
            tv.utils.save_image(grid.float(), os.path.join(evaluate_path, f"evaluate{(self.step+self.resume_step):06d}_{j}.png"), normalize=True)
        logger.log(f"create{len(all_samples)*self.batch_size} samples")

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['objs'] = data['objs'].long()
        data['boxes'] = data['boxes'].long()
        # no need to do the label embedding here
        if self.drop_rate > 0.0:
            mask_objs = (th.rand([data['objs'].shape[0], 1, 1, 1]) > self.drop_rate).float()    # 8 is the num_objs
            input_objs = data['objs'] * mask_objs
            mask_boxes = (th.rand([data['boxes'].shape[0], 1, 1, 1]) > self.drop_rate).float()
            input_boxes = data['boxes'] * mask_boxes
        else:
            input_objs = data['objs']
            input_boxes = data['boxes']

        cond = {key: value for key, value in data.items() if key not in ['objs', 'boxes', 'instance', 'path', 'label_ori']}
        cond['objs'] = input_objs
        cond['boxes'] = input_boxes

        return cond


    def get_edges(self, t):
        edge = th.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

def bbox_preprocessing(image_boxes):    # to preprocess the bounding boxes for non-existance
    image_size = 128
    for idx, obj_box in enumerate(image_boxes):
        for value in obj_box:
            if value < 0:
                 image_boxes[idx] = th.zeros_like(image_boxes[idx])
        obj_box[0] = int(obj_box[0]*image_size)
        obj_box[1] = int(obj_box[1]*image_size)
        obj_box[2] = obj_box[0] + int(obj_box[2]*image_size)
        obj_box[3] = obj_box[1] + int(obj_box[3]*image_size)
    return image_boxes

def bbox_preprocessing_SDI(image_boxes):    # to preprocess the bounding boxes for non-existance
    image_size = 128
    for idx, obj_box in enumerate(image_boxes):
        for value in obj_box:
            if value < 0:
                 image_boxes[idx] = th.zeros_like(image_boxes[idx])
        obj_box[0] = int(obj_box[0]*image_size)
        obj_box[1] = int(obj_box[1]*image_size)
        obj_box[2] = int(obj_box[2]*image_size)
        obj_box[3] = int(obj_box[3]*image_size)
    return image_boxes

def image_preprocessing(image): #to preprocess image to denoirmalize into uint8
    #denorm = tv.transforms.Normalize(mean=[-1, -1, -1], std=[2, ,2 ,2])
    #image = denorm(image)*255
    image = ((image+1)*127.5).to(th.int8)

    return image

def sample_preprocessing(image): #to preprocess image to denoirmalize into uint8
    #denorm = tv.transforms.Normalize(mean=[-1, -1, -1], std=[2, 2 ,2])
    #image = denorm(image)*255
    image = ((image+1)*255).to(th.int8)

    return image

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
