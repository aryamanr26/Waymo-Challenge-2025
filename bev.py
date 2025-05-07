import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVFeatureEncoder(nn.Module):
    """
    Explicit BEV Feature Encoding:
    - Input: Q-former queries for each view, shape [B, V*M, D].
    - Output: BEV tokens, shape [B, X*Y, D].
    """
    def __init__(
        self,
        num_views: int,
        num_queries_per_view: int,
        query_dim: int,
        bev_dim: int,
        bev_h: int,
        bev_w: int,
        dequery_hidden_dim: int = None,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_queries_per_view = num_queries_per_view
        self.query_dim = query_dim
        self.bev_dim = bev_dim
        self.bev_h = bev_h
        self.bev_w = bev_w # De-query head: maps flattened queries per view to BEV feature map
        hidden_dim = dequery_hidden_dim or (bev_dim * bev_h * bev_w)
        self.dequery_head = nn.Sequential(
            nn.Linear(query_dim * num_queries_per_view, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bev_dim * bev_h * bev_w),
        ) # Learnable homographies per view (3x3)
        self.register_parameter(
            "homographies",
            nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_views,1,1))
        ) # Linear projection to query_dim (text/planner embedding)
        self.proj = nn.Linear(bev_dim, query_dim)

    def warp_feature(self, feat: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        """
        Warp feature map into BEV grid using homography.
        feat: [B, bev_dim, bev_h, bev_w]
        homography: [3,3]
        returns: [B, bev_dim, bev_h, bev_w]
        """
        B, C, H, W = feat.shape
        device = feat.device # create BEV grid coords
        ys, xs = torch.meshgrid(
            torch.linspace(0, H-1, H, device=device),
            torch.linspace(0, W-1, W, device=device),
            indexing='ij'
        )
        ones = torch.ones_like(xs)
        coords = torch.stack([xs.flatten(), ys.flatten(), ones.flatten()], dim=0)  # [3, H*W] # compute source coords
        hom_inv = torch.inverse(homography)
        src = hom_inv @ coords  # [3, N]
        src = src[:2] / src[2:3]
        # Normalize to [-1,1]
        src_x = (src[0].view(H, W) / (W - 1) * 2 - 1)
        src_y = (src[1].view(H, W) / (H - 1) * 2 - 1)
        grid = torch.stack((src_x, src_y), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        return F.grid_sample(feat, grid, align_corners=True)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        queries: [B, V*M, D]
        returns: bev_tokens [B, bev_h*bev_w, D]
        """
        B, QM, D = queries.shape
        assert QM == self.num_views * self.num_queries_per_view # reshape queries per view
        q = queries.reshape(B, self.num_views, self.num_queries_per_view * D)
        # q = queries.view(B, self.num_views, self.num_queries_per_view * D)  # [B, V, M*D]
        print(f"Reshaped queries: {q.shape}")
        bev_sum = torch.zeros(B, self.bev_dim, self.bev_h, self.bev_w, device=queries.device)
        for v in range(self.num_views):
            qv = q[:, v]  # [B, M*D]
            feat = self.dequery_head(qv).view(B, self.bev_dim, self.bev_h, self.bev_w)
            print(f"Feature map for view {v}: {feat.shape}")
            H = self.homographies[v]
            warped = self.warp_feature(feat, H)
            print(f"Warped feature for view {v}: {warped.shape}")
            bev_sum += warped # flatten and project
        print(f"Summed BEV features: {bev_sum.shape}")
        bev_flat = bev_sum.view(B, self.bev_dim, -1).permute(0, 2, 1)  # [B, bev_h*bev_w, bev_dim]
        print(f"Flattened BEV grid: {bev_flat.shape}")
        bev_tokens = self.proj(bev_flat)  # [B, bev_h*bev_w, query_dim]
        print(f"BEV tokens after projection: {bev_tokens.shape}")
        return bev_tokens

# === Test snippet ===
if __name__ == "__main__":
    B, V, C, H, W = 4, 8, 3, 224, 224
    M, D = 16, 1024
    bev_dim, bev_h, bev_w = 64, 16, 16

    images = torch.randn(B, V, C, H, W)
    visual_tokens = torch.randn(B, V * M, D)

    print("Images:", images.shape)               # torch.Size([4, 8, 3, 224, 224])
    print("VisualTokens:", visual_tokens.shape)  # torch.Size([4, 128, 1024])

    encoder = BEVFeatureEncoder(
        num_views=V,
        num_queries_per_view=M,
        query_dim=D,
        bev_dim=bev_dim,
        bev_h=bev_h,
        bev_w=bev_w,
    )
    bev_tokens = encoder(visual_tokens)
    print("Final BEV tokens:", bev_tokens.shape)  # torch.Size([4, 256, 1024])