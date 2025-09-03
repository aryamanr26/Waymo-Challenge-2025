import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import warnings
import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

accelerator = Accelerator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch using:", device)


from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_dataset_loader import WaymoDatasetLoader
from waymo_E2EDataset import WaymoE2EDataset

from vision_encoder import MultiViewQFormer
from bev import BEVFeatureEncoder
from tfusion import TemporalFusion
from embedding import PoseTokenEncoder, RouteTokenEncoder
from planner_head import PlannerHead4D
from depth_est import DepthEstimator, DepthFusionEmbedder


def compute_challenge_loss(pred, targ):
    """
    pred: (B, T, P)
    targ: (B, T, Q)
    We’ll use the first D = min(P,Q) dims of each.
    """
    # determine how many dims to compare (e.g. 2 for x,y)
    D = min(pred.shape[-1], targ.shape[-1])
    pred2 = pred[..., :D]
    targ2 = targ[..., :D]

    # per-timestep L2
    dists = torch.norm(pred2 - targ2, dim=-1)  # (B, T)
    # ADE@3s (first 12 at 4 Hz)
    ade3 = dists[:, :12].mean(dim=1).mean()
    # ADE@5s (all 20 steps)
    ade5 = dists.mean(dim=1).mean()
    loss = ade3 + ade5
    return loss, ade3, ade5

def save_checkpoint(epoch, model_components, optimizer, loss, filename="checkpoints/checkpoint.pth"):
    model_state = {}
    for name, module in model_components.items():
        if hasattr(module, 'state_dict'):
            model_state[name] = module.state_dict()
        else:
            # skip wrappers like your DepthEstimator
            print(f"  [checkpoint] skipping `{name}` (no state_dict)")
    torch.save({
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }, filename)


def load_checkpoint(filename, model_components, optimizer=None, map_location=None):
    ckpt = torch.load(filename, map_location=map_location or device)
    for name, module in model_components.items():
        if name in ckpt['model_state']:
            module.load_state_dict(ckpt['model_state'][name])
        else:
            print(f"  [checkpoint] no saved state for `{name}`, skipping")
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt['epoch'], ckpt['loss']

if __name__ == "__main__":
    # Dataset
    loader = WaymoDatasetLoader()
    train_files, _, _ = loader.get_file_lists()
    builder = WaymoE2EDataset(batch_size=8)
    train_ds = builder.build_dataset(train_files)

    # Models
    V, M, NM, D = 8, 16, 128, 1024
    bev_dim, bev_h, bev_w = 64, 16, 16

    qformer   = MultiViewQFormer(V, M, "openai/clip-vit-large-patch14", 2, 16).to(device)
    encoder   = BEVFeatureEncoder(V, M, D, bev_dim, bev_h, bev_w).to(device)
    tf_module = TemporalFusion(NM, D).to(device)
    pose_enc  = PoseTokenEncoder(64, D).to(device)
    route_enc = RouteTokenEncoder(16, D).to(device)
    planner   = PlannerHead4D("google/flan-t5-large", D).to(device)
    DPT       = DepthEstimator()
    depth_embedder = DepthFusionEmbedder(
        patch_size=16,
        embed_dim=1024,
        n_tokens=256,
        add_positional=True,
        project_to=1024  # or whatever your LLM expects
    ).to(device)

    _ = torch.randn(1, device='cuda') * 2  # warm up CUDA context
    torch.cuda.synchronize()  # ensure the operation finishes
    print("CUDA context warmed up.")

    # Accelerate
    qformer, encoder, tf_module, pose_enc, route_enc, planner, DPT = accelerator.prepare(
        qformer, encoder, tf_module, pose_enc, route_enc, planner, DPT
    )

    # Optimizer
    optimizer = optim.AdamW([
        *qformer.parameters(),
        *encoder.parameters(),
        *tf_module.parameters(),
        *pose_enc.parameters(),
        *route_enc.parameters(),
        *planner.parameters(),
    ], lr=1e-4)
    optimizer = accelerator.prepare(optimizer)

    # Bundle for checkpoint
    model_components = {
        'qformer': qformer,
        'encoder': encoder,
        'tf_module': tf_module,
        'pose':    pose_enc,
        'route':   route_enc,
        'planner': planner,
        #'depth':   DPT
    }

    resume = None  # e.g. "checkpoints/checkpoint_epoch_5.pth"
    if resume:
        start_ep, _ = load_checkpoint(resume, model_components, optimizer)
    else:
        start_ep = 0

    # Checkpoint dir
    ckpt_dir = "checkpoints3"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    EPOCHS = 10
    for epoch in range(start_ep, EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        total_loss = 0.0

        for it, batch in enumerate(train_ds):
            # start = time.time()
            images, intent, past, future, pose_tok, route_tok = batch
            images    = torch.from_numpy(images.numpy()).to(device)
            pose_tok  = torch.from_numpy(pose_tok.numpy()).to(device)
            future    = torch.from_numpy(future.numpy())[..., :2].to(device)  # only x,y
            route_tok = torch.from_numpy(route_tok.numpy()).to(device)

            optimizer.zero_grad()

            # Depth per-sample
            # dim = images.permute(0,1,4,2,3)
            # depths = [DPT.estimate_depth_batch(dim[i:i+1]) for i in range(dim.size(0))]
            # depth_maps = torch.cat(depths,0).unsqueeze(2).repeat(1,1,3,1,1).to(device)
            
            # print(depth_maps.shape)
            
            depth_maps = DPT.estimate_depth_batch_torch(images)
            d_emb = depth_embedder(depth_maps)
            # print(d_emb.shape)
            # (8, 256, 1024)
	    
            # Forward
            # d_toks = qformer(depth_maps)
            # d_emb  = encoder(d_toks)
            i_toks = qformer(images.permute(0,1,4,2,3).to(device))
            bev    = encoder(i_toks)
            t_emb  = tf_module(i_toks)
            p_emb  = pose_enc(pose_tok)
            r_emb  = route_enc(route_tok)

            means = []
            for b in range(bev.size(0)):
                m, _ = planner(bev[b:b+1], t_emb[b:b+1], p_emb[b:b+1], r_emb[b:b+1], d_emb[b:b+1])
                means.append(m)
            means = torch.cat(means, 0)  # (B, 20, 2)

            # Compute challenge loss & metrics
            loss, ade3, ade5 = compute_challenge_loss(means, future)
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            # end = time.time() - start
            # print(f"Time elapsed: {end:.2f} seconds")
            if it % 100 == 0:
                print(f" iter {it:4d}  LOSS: {loss.item():.4f}  ADE3: {ade3:.4f}  ADE5: {ade5:.4f}")

        print(f"Epoch {epoch+1} summary loss: {total_loss:.4f}")

        # Save checkpoint
        path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(epoch+1, model_components, optimizer, total_loss, filename=path)
        print(f" Saved checkpoint: {path}")

    print("\nTraining complete—all checkpoints in", ckpt_dir)

