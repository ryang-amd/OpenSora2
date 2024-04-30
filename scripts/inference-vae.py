import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets import prepare_dataloader, save_sample
from opensora.models.vae.losses import VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    if os.environ.get("WORLD_SIZE", None):
        use_dist = True
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
    else:
        use_dist = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = build_module(cfg.dataset, DATASETS)
    dataloader = prepare_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    print(f"Dataset contains {len(dataset):,} videos ({cfg.dataset.data_path})")
    total_batch_size = cfg.batch_size * dist.get_world_size()
    print(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model & load weights
    # ======================================================
    # 4.1. build model
    model = build_module(cfg.model, MODELS)
    model.to(device, dtype).eval()

    # ======================================================
    # 5. inference
    # ======================================================
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # define loss function
    vae_loss_fn = VAELoss(
        logvar_init=cfg.get("logvar_init", 0.0),
        perceptual_loss_weight=cfg.perceptual_loss_weight,
        kl_loss_weight=cfg.kl_loss_weight,
        device=device,
        dtype=dtype,
    )

    # get total number of steps
    total_steps = len(dataloader)
    if cfg.max_test_samples is not None:
        total_steps = min(int(cfg.max_test_samples // cfg.batch_size), total_steps)
        print(f"limiting test dataset to {int(cfg.max_test_samples//cfg.batch_size) * cfg.batch_size}")
    dataloader_iter = iter(dataloader)

    running_loss = running_nll = 0.0
    loss_steps = 0
    with tqdm(
        range(total_steps),
        disable=not coordinator.is_master(),
        total=total_steps,
        initial=0,
    ) as pbar:
        for step in pbar:
            batch = next(dataloader_iter)
            x = batch["video"].to(device, dtype)  # [B, C, T, H, W]

            #  ===== VAE =====
            z, posterior, x_z = model.encode(x, training=True)
            x_rec, _ = model.decode(z, num_frames=x.size(2))
            x_ref = model.spatial_vae.decode(x_z)

            # loss calculation
            nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)
            vae_loss = weighted_nll_loss + weighted_kl_loss
            loss_steps += 1
            running_loss = vae_loss.item() / loss_steps + running_loss * ((loss_steps - 1) / loss_steps)
            running_nll = nll_loss.item() / loss_steps + running_nll * ((loss_steps - 1) / loss_steps)

            if not use_dist or coordinator.is_master():
                ori_dir = f"{save_dir}_ori"
                rec_dir = f"{save_dir}_rec"
                ref_dir = f"{save_dir}_ref"
                os.makedirs(ori_dir, exist_ok=True)
                os.makedirs(rec_dir, exist_ok=True)
                os.makedirs(ref_dir, exist_ok=True)
                for idx, vid in enumerate(x):
                    pos = step * cfg.batch_size + idx
                    save_sample(vid, fps=cfg.fps, save_path=f"{ori_dir}/{pos:03d}")
                    save_sample(x_rec[idx], fps=cfg.fps, save_path=f"{rec_dir}/{pos:03d}")
                    save_sample(x_ref[idx], fps=cfg.fps, save_path=f"{ref_dir}/{pos:03d}")

    print("test vae loss:", running_loss)
    print("test nll loss:", running_nll)


if __name__ == "__main__":
    main()
