"""
reasoning_module.py

SGG 推理层基础设施
支持的推理架构（5种，对应论文实验变量）:

  GNN 家族（用于验证拉普拉斯平滑与身份湮灭）:
    - gcn        经典图卷积，对称归一化 D^{-1/2} A D^{-1/2}
    - gat        图注意力网络，多头注意力聚合（验证注意力塌缩在图域的表现）
    - gine       GIN + 边特征融合（验证边特征是否缓解身份湮灭）
    - gated_gcn  门控图卷积（验证门控机制对过平滑的抑制）

  Transformer 家族（用于验证注意力塌缩）:
    - transformer  标准 Multi-Head Self-Attention + FFN

所有层均内置特征探针接口，由 ReasoningInfrastructure 统一调用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# ===========================================================================
# [Layer 1] GCN — 基线：验证拉普拉斯平滑
# ===========================================================================

class GCNLayer(nn.Module):
    """
    标准图卷积层（Kipf & Welling 2017）
    对称归一化: Â = D^{-1/2} (A+I) D^{-1/2}
    用于实验：量化纯聚合操作导致的身份湮灭速率
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 加自环：A_hat = A + I
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj_hat.sum(1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        norm_adj = d_inv_sqrt.unsqueeze(1) * adj_hat * d_inv_sqrt.unsqueeze(0)

        out = self.proj(torch.mm(norm_adj, x))
        out = self.norm(out)
        return F.relu(out)


# ===========================================================================
# [Layer 2] GAT — 图注意力网络：比较注意力塌缩在图域的表现
# ===========================================================================

class GATLayer(nn.Module):
    """
    图注意力层（Veličković et al. 2018）多头版本
    注意力系数: e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
    实验价值：
      - GAT 的注意力权重是在图结构内学习的，与 Transformer 的全局注意力对比
      - 诊断：GAT 是否也会产生注意力塌缩（所有权重趋向均匀）
      - 探针：last_attn_weights 供 get_attention_collapse_score() 使用
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim 必须被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads

        # 节点特征变换（每个头独立）
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # 注意力向量 a：每个头对应 2*head_dim
        self.a = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(out_dim)

        # 探针：[num_heads, N, N]
        self.last_attn_weights: torch.Tensor = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        Wh = self.W(x).view(N, self.num_heads, self.head_dim)  # [N, H, D_h]

        # 计算注意力分数 e_{ij}（向量化实现）
        # [N, H, D_h] → 拼接 source 和 target
        Wh_i = Wh.unsqueeze(1).expand(N, N, self.num_heads, self.head_dim)  # [N, N, H, D_h]
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.num_heads, self.head_dim)  # [N, N, H, D_h]
        pair = torch.cat([Wh_i, Wh_j], dim=-1)                               # [N, N, H, 2D_h]

        # e = LeakyReLU(pair · a)
        e = self.leaky_relu(
            (pair * self.a.unsqueeze(0).unsqueeze(0)).sum(-1)                 # [N, N, H]
        )

        # Keep graph edges plus self-loops. Without explicit self-loops an
        # isolated relation token would otherwise fall back to attending every
        # token after NaN handling, silently turning a sparse graph into a
        # complete graph.
        adj_hat = (adj > 0) | torch.eye(N, dtype=torch.bool, device=adj.device)
        mask = (~adj_hat).unsqueeze(-1).expand_as(e)    # [N, N, H]
        e = e.masked_fill(mask, float('-inf'))

        # softmax → dropout
        alpha = torch.softmax(e, dim=1)                  # [N, N, H]
        # 处理全 -inf 行（孤立节点），避免 NaN
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # 保存探针（取各头均值用于塌缩诊断）
        self.last_attn_weights = alpha.mean(dim=-1).detach()  # [N, N]

        # 聚合
        # alpha: [N, N, H], Wh: [N, H, D_h]
        out = torch.einsum('ijh,jhd->ihd', alpha, Wh).reshape(N, -1)
        # [N, H*D_h] = [N, out_dim]

        out = self.norm(out + self.W(x) if x.size(-1) == out.size(-1) else out)
        return F.elu(out)


# ===========================================================================
# [Layer 3] GINE — GIN + 边特征融合
# ===========================================================================

class GINELayer(nn.Module):
    """
    GIN with Edge features（Hu et al. 2020）
    聚合函数: h_i' = MLP((1+ε) h_i + Σ_{j∈N(i)} ReLU(h_j + e_{ij}))
    
    边特征 e_{ij} 此处用 adj 矩阵的权值（若为二值图则退化为 GIN）
    实验价值：验证边特征注入是否能缓解节点身份湮灭
    """
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        # MLP：两层，中间加 BN
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        # 边特征投影：将标量权值映射到节点特征维度
        self.edge_proj = nn.Linear(1, in_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)

        # 边特征：将 adj 权值 [N, N, 1] 投影到 [N, N, in_dim]
        edge_feat = self.edge_proj(adj.unsqueeze(-1))   # [N, N, in_dim]

        # 逐边聚合：对每条边，h_j + e_{ij}，再按行求和
        # [N, N, in_dim] + [N, in_dim].unsqueeze(0) → [N, N, in_dim]
        neighbor_msg = F.relu(x.unsqueeze(0) + edge_feat)  # [N, N, in_dim]
        # adj 掩码：只聚合有边的邻居
        mask = adj.unsqueeze(-1).float()
        agg = (neighbor_msg * mask).sum(dim=1)              # [N, in_dim]

        out = self.mlp((1 + self.eps) * x + agg)
        return self.norm(out)


# ===========================================================================
# [Layer 4] GatedGCN — 门控图卷积（缓解过平滑）
# ===========================================================================

class GatedGCNLayer(nn.Module):
    """
    门控图卷积（Bresson & Laurent 2017）
    引入边门控向量 e_{ij}，动态控制每条边对目标节点的贡献比例
    
    更新规则:
      η_{ij} = sigmoid(U·h_i + V·h_j + b_e)   # 边门控
      h_i'  = h_i + ReLU(BN(W·h_i + Σ_j η_{ij} ⊙ A·h_j))
    
    实验价值：
      - 门控向量提供每条边的选择性抑制能力
      - 验证门控机制能否在多层传播中阻止特征均匀化
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # 节点变换
        self.W_h  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_x  = nn.Linear(in_dim, out_dim, bias=False)
        # 边门控：U(h_i) + V(h_j)
        self.U    = nn.Linear(in_dim, in_dim, bias=False)
        self.V    = nn.Linear(in_dim, in_dim, bias=False)
        self.bn   = nn.LayerNorm(out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)

        # 边门控系数 η_{ij}: [N, N, in_dim]
        gate = torch.sigmoid(
            self.U(x).unsqueeze(1) + self.V(x).unsqueeze(0)   # broadcast: [N, N, in_dim]
        )

        # Apply the learned gate to each neighbour representation before
        # aggregation. Summing gate values alone discards neighbour identity.
        gated_neighbors = (
            gate * x.unsqueeze(0) * adj.unsqueeze(-1)
        ).sum(dim=1)
        message = self.W_x(gated_neighbors)
        centre = self.W_h(x)

        # 残差融合
        if x.size(-1) == message.size(-1):
            out = x + F.relu(self.bn(centre + message))
        else:
            out = F.relu(self.bn(centre + message))

        return self.norm(out)


# ===========================================================================
# [Layer 5] Transformer — 验证注意力塌缩
# ===========================================================================

class TransformerLayer(nn.Module):
    """
    标准 Post-LN Transformer 层
    保存 last_attn_weights 供 get_attention_collapse_score() 诊断
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.mha   = nn.MultiheadAttention(d_model, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.last_attn_weights: torch.Tensor = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x_3d = x.unsqueeze(0)   # [1, N, D]

        # 构建布尔掩码（无边位置为 True，被忽略）
        bool_mask = None
        if attn_mask is not None:
            allowed = (attn_mask > 0) | torch.eye(
                attn_mask.size(0), dtype=torch.bool, device=attn_mask.device
            )
            bool_mask = ~allowed

        attn_out, attn_w = self.mha(x_3d, x_3d, x_3d, attn_mask=bool_mask)
        self.last_attn_weights = attn_w.detach()           # [1, N, N]

        x_3d = self.norm1(x_3d + attn_out)
        x_3d = self.norm2(x_3d + self.ffn(x_3d))
        return x_3d.squeeze(0)                             # [N, D]

    def get_attention_spectrum(self) -> torch.Tensor:
        if self.last_attn_weights is None:
            return None
        W = self.last_attn_weights.squeeze(0)
        with torch.no_grad():
            return torch.linalg.svdvals(W)


# ===========================================================================
# 模块级诊断接口
# ===========================================================================

def get_layer_rank(features: torch.Tensor, threshold: float = 0.01) -> int:
    with torch.no_grad():
        s = torch.linalg.svdvals(features.float())
        s_norm = s / (s.sum() + 1e-12)
        cumulative = torch.cumsum(s_norm, dim=0)
        rank = int((cumulative < (1.0 - threshold)).sum().item()) + 1
        return min(rank, min(features.shape))


def get_dirichlet_energy(features: torch.Tensor, adj: torch.Tensor) -> float:
    with torch.no_grad():
        N = features.size(0)
        deg = torch.diag(adj.sum(1))
        L = (deg - adj).float()
        f = features.float()
        energy = torch.trace(f.t() @ L @ f)
        return (energy / N).item()


def get_attention_collapse_score(attn_weights: torch.Tensor) -> dict:
    """适用于 TransformerLayer 和 GATLayer 的 last_attn_weights"""
    with torch.no_grad():
        W = attn_weights.squeeze(0) if attn_weights.dim() == 3 else attn_weights
        S = torch.linalg.svdvals(W.float())
        s_norm = S / (S.sum() + 1e-12)
        dominant_ratio = s_norm[0].item()
        effective_rank = get_layer_rank(W)
    return {
        'effective_rank':        effective_rank,
        'dominant_singular_ratio': dominant_ratio,
        'singular_values':       S.cpu().numpy(),
    }

# ===========================================================================
# 核心调度类
# ===========================================================================

class ReasoningInfrastructure(nn.Module):
    """
    推理层工厂：根据 mode 字符串实例化对应的层堆叠
    统一接口：forward(x, adj) → logits
    内置特征探针 feature_probes，供病理诊断模块逐层读取
    """

    MODE_REGISTRY = {
        'gcn':         GCNLayer,
        'gat':         GATLayer,
        'gine':        GINELayer,
        'gated_gcn':   GatedGCNLayer,
        'transformer': TransformerLayer,
    }

    def __init__(
        self,
        input_dim:  int = 768,
        hidden_dim: int = 512,
        num_layers: int = 3,
        mode:       str = 'gcn',
        num_classes: int = 50,
        # GAT 专用参数
        gat_heads:  int = 4,
        gat_dropout: float = 0.1,
        # Transformer 专用参数
        transformer_heads: int = 8,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        mode = mode.lower()
        if mode not in self.MODE_REGISTRY:
            raise ValueError(
                f"Unknown reasoning mode '{mode}'. "
                f"Available: {list(self.MODE_REGISTRY.keys())}"
            )
        self.mode = mode
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.feature_probes = []   # 每次 forward 后由探针填充

        # 根据 mode 构造层堆叠
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if mode == 'gcn':
                self.layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif mode == 'gat':
                self.layers.append(
                    GATLayer(hidden_dim, hidden_dim,
                             num_heads=gat_heads, dropout=gat_dropout)
                )
            elif mode == 'gine':
                self.layers.append(GINELayer(hidden_dim, hidden_dim))
            elif mode == 'gated_gcn':
                self.layers.append(GatedGCNLayer(hidden_dim, hidden_dim))
            elif mode == 'transformer':
                self.layers.append(
                    TransformerLayer(hidden_dim,
                                     num_heads=transformer_heads,
                                     dropout=transformer_dropout)
                )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """
        x:   [N, input_dim] 节点特征
        adj: [N, N] 邻接矩阵（GNN 模式必需）
        返回: [N, num_classes] 谓词 logits
        """
        self.feature_probes = []

        out = self.input_proj(x)
        self.feature_probes.append(out.detach())   # probe[0]: 投影后原始特征

        for layer in self.layers:
            if self.mode == 'transformer':
                out = layer(out, adj)
            else:
                if adj is None:
                    raise ValueError(f"Mode '{self.mode}' requires adjacency matrix.")
                out = layer(out, adj)
            self.feature_probes.append(out.detach())

        return self.classifier(out)

    @classmethod
    def list_modes(cls):
        print(f"Available reasoning modes: {list(cls.MODE_REGISTRY.keys())}")
