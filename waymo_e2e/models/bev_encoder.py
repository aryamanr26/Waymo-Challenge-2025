import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVFeatureEncoder(nn.Module):
    """
    Map multi-view Q-Former tokens to a coarse BEV grid via per-view MLP + homography warp + sum.
    """

    def __init__(
        self,
        num_views: int,
        num_queries_per_view: int,
        query_dim: int,
        bev_dim: int,
        bev_h: int,
        bev_w: int,
        dequery_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_queries_per_view = num_queries_per_view
        self.query_dim = query_dim
        self.bev_dim = bev_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        hidden_dim = dequery_hidden_dim or (bev_dim * bev_h * bev_w)
        self.dequery_head = nn.Sequential(
            nn.Linear(query_dim * num_queries_per_view, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bev_dim * bev_h * bev_w),
        )
        self.register_parameter("homographies", nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_views, 1, 1)))
        self.proj = nn.Linear(bev_dim, query_dim)

    def warp_feature(self, feat: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        device = feat.device
        ys, xs = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=device),
            torch.linspace(0, W - 1, W, device=device),
            indexing="ij",
        )
        ones = torch.ones_like(xs)
        coords = torch.stack([xs.flatten(), ys.flatten(), ones.flatten()], dim=0)
        hom_inv = torch.inverse(homography)
        src = hom_inv @ coords
        src = src[:2] / src[2:3]
        src_x = src[0].view(H, W) / (W - 1) * 2 - 1
        src_y = src[1].view(H, W) / (H - 1) * 2 - 1
        grid = torch.stack((src_x, src_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return F.grid_sample(feat, grid, align_corners=True)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        B, QM, D = queries.shape
        assert QM == self.num_views * self.num_queries_per_view
        q = queries.reshape(B, self.num_views, self.num_queries_per_view * D)
        bev_sum = torch.zeros(B, self.bev_dim, self.bev_h, self.bev_w, device=queries.device)
        for v in range(self.num_views):
            qv = q[:, v]
            feat = self.dequery_head(qv).view(B, self.bev_dim, self.bev_h, self.bev_w)
            H = self.homographies[v]
            warped = self.warp_feature(feat, H)
            bev_sum += warped
        bev_flat = bev_sum.view(B, self.bev_dim, -1).permute(0, 2, 1)
        return self.proj(bev_flat)
