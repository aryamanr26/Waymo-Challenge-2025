import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Your modules
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset
from vision_encoder import MultiViewQFormer
from bev import BEVFeatureEncoder
from tfusion import TemporalFusion
from embedding import PoseTokenEncoder, RouteTokenEncoder
from planner_head import PlannerHead4D
from depth_est import DepthEstimator, DepthFusionEmbedder

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def compute_challenge_loss(pred, targ):
    D = min(pred.shape[-1], targ.shape[-1])
    pred2 = pred[..., :D]
    targ2 = targ[..., :D]
    dists = torch.norm(pred2 - targ2, dim=-1)
    ade3 = dists[:, :12].mean(dim=1).mean()
    ade5 = dists.mean(dim=1).mean()
    loss = ade3 + ade5
    return loss, ade3, ade5


def train(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    print(f"[Rank {rank}] Running on {device}")

    # Dataset
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    dataset = WaymoE2EDataset()

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dl = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True)

    # Model
    V, M, NM, D = 8, 16, 128, 1024
    bev_dim, bev_h, bev_w = 64, 16, 16
    qformer   = MultiViewQFormer(V, M, "openai/clip-vit-large-patch14", 2, 16).to(device)
    encoder   = BEVFeatureEncoder(V, M, D, bev_dim, bev_h, bev_w).to(device)
    tf_module = TemporalFusion(NM, D).to(device)
    pose_enc  = PoseTokenEncoder(64, D).to(device)
    route_enc = RouteTokenEncoder(16, D).to(device)
    planner   = PlannerHead4D("google/flan-t5-large", D).to(device)
    DPT       = DepthEstimator().to(device)
    depth_embedder = DepthFusionEmbedder(patch_size=16, embed_dim=1024, n_tokens=256,
                                         add_positional=True, project_to=1024).to(device)

    # Wrap with DDP
    qformer   = DDP(qformer, device_ids=[rank])
    encoder   = DDP(encoder, device_ids=[rank])
    tf_module = DDP(tf_module, device_ids=[rank])
    pose_enc  = DDP(pose_enc, device_ids=[rank])
    route_enc = DDP(route_enc, device_ids=[rank])
    planner   = DDP(planner, device_ids=[rank])
    # DPT not wrapped (no training)

    optimizer = optim.AdamW([
        *qformer.parameters(),
        *encoder.parameters(),
        *tf_module.parameters(),
        *pose_enc.parameters(),
        *route_enc.parameters(),
        *planner.parameters(),
    ], lr=1e-4)

    # Training
    EPOCHS = 10
    for epoch in range(EPOCHS):
        train_dl.sampler.set_epoch(epoch)
        total_loss = 0.0

        for it, batch in enumerate(train_dl):
            images, intent, past, future, pose_tok, route_tok = batch
            images    = images.to(device)
            pose_tok  = pose_tok.to(device)
            future    = future[..., :2].to(device)
            route_tok = route_tok.to(device)

            optimizer.zero_grad()

            depth_maps = DPT.estimate_depth_batch_torch(images)
            d_emb = depth_embedder(depth_maps)

            i_toks = qformer(images.permute(0,1,4,2,3))
            bev    = encoder(i_toks)
            t_emb  = tf_module(i_toks)
            p_emb  = pose_enc(pose_tok)
            r_emb  = route_enc(route_tok)

            means = []
            for b in range(bev.size(0)):
                m, _ = planner(bev[b:b+1], t_emb[b:b+1], p_emb[b:b+1], r_emb[b:b+1], d_emb[b:b+1])
                means.append(m)
            means = torch.cat(means, 0)

            loss, ade3, ade5 = compute_challenge_loss(means, future)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if it % 100 == 0 and rank == 0:
                print(f"[Rank {rank}] Iter {it} | Loss: {loss.item():.4f} | ADE3: {ade3:.4f} | ADE5: {ade5:.4f}")

        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
            torch.save({
                'epoch': epoch + 1,
                'qformer': qformer.module.state_dict(),
                'encoder': encoder.module.state_dict(),
                'planner': planner.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f"checkpoints3/checkpoint_epoch_{epoch+1}.pth")

    cleanup()

