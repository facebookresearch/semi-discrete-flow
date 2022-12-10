import torch
import torch.nn.functional as F

from survae.transforms import Softplus
from survae.transforms.surjections import Surjection

class VoronoiSurjection(Surjection):
    """
    A generative argmax surjection using a Cartesian product of binary spaces. Argmax is performed over the final dimension.
    Args:
        encoder: ConditionalDistribution, a distribution q(z|x) with support over z s.t. x=argmax z.
    Example:
        Input tensor x of shape (B, D, L) with discrete values {0,1,...,C-1}:
        encoder should be a distribution of shape (B, D, L, D), where D=ceil(log2(C)).
        When e.g. C=27, we have D=5, such that 2**5=32 classes are represented.
    """

    stochastic_forward = True

    def __init__(self, noise_dist, voronoi_transform, num_classes, embedding_dim):
        super().__init__()

        self.noise_dist = noise_dist
        self.voronoi_transform = voronoi_transform
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.softplus = Softplus()

    def forward(self, x):
        B, L = x.shape[0], x.shape[2]
        N, K, D = 1, self.num_classes, self.embedding_dim

        u, log_pu = self.noise_dist.sample_with_log_prob(context=x)
        u = u.reshape(B * L, 1, D)

        mask = F.one_hot(x, self.num_classes).bool().to(x.device)  # (B, 1, L, K)
        mask = mask.reshape(B * L, 1, K)

        # Center the flow at the Voronoi cell.
        points = self.voronoi_transform.anchor_pts.reshape(1, N, K, D)
        x_k = torch.masked_select(points, mask.reshape(-1, N, K, 1)).reshape(-1, N, D)
        z = u + x_k

        # Transform into the target Voronoi cell.
        z, ldj = self.voronoi_transform.map_onto_cell(z, mask=mask)

        z = z.reshape(B, 1, L, D)
        ldj = ldj.reshape(B, -1)

        log_qz = log_pu - ldj.sum(1)

        ldj = -log_qz
        return z, ldj

    def inverse(self, z):
        raise NotImplementedError