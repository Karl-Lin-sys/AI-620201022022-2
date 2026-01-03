"""
🧬 ProteinAI - タンパク質のカタチを理解するモジュール (GCN)
===========================================================

【やさしい説明】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

タンパク質は、アミノ酸が鎖のように繋がっていますが、
実際は「折りたたまれた」複雑な形をしています。

レゴブロックを想像してください。
ブロックを繋げていくと、ただの棒ではなく、
ロボットや家のような形になりますよね。
タンパク質も同じように、折りたたまれて特別な形になります。

このプログラムは「グラフニューラルネットワーク（GCN）」を使って、
タンパク質の「繋がり方」と「形」を理解します。

【技術的説明】
- GCN（Graph Convolutional Network）: 分子をグラフとして表現し学習
- 各アミノ酸がノード（点）、結合がエッジ（線）
- 隣接するアミノ酸から情報を集めて特徴を学習

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


class GraphConvLayer(nn.Module):
    """
    グラフ畳み込み層
    
    【やさしい説明】
    友達の輪を想像してください。
    あなたの周りにいる友達が「元気だよ！」と言ったら、
    あなたも「みんな元気なんだな」と分かりますよね。
    
    この層は、周りのアミノ酸から情報を集めて、
    「このアミノ酸はどんな環境にいるのか」を理解します。
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 周りのノードから情報を集める
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support) + self.bias
        return output


class ProteinGCN(nn.Module):
    """
    タンパク質構造解析GCN
    
    【やさしい説明】
    タンパク質を「人間関係図」のように表現します。
    
    - 各アミノ酸 = 一人の人
    - 近くにいる = 友達（繋がりがある）
    
    GCNは「友達の友達」まで情報を集めて、
    「このアミノ酸は構造の中でどんな役割？」を理解します。
    
    最終的に「このタンパク質はαヘリックス構造かな？」
    「βシート構造かな？」と予測できます。
    """
    
    def __init__(self, node_features: int = 64, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.encoder = nn.Linear(node_features, hidden)
        self.gcn1 = GraphConvLayer(hidden, hidden)
        self.gcn2 = GraphConvLayer(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes)
        )
        self.structure_names = [
            'αヘリックス', 'βシート', 'ループ', 'ターン', 'コイル',
            'ヘリックス-ターン-ヘリックス', 'βバレル', 'α/β混合', '全αタイプ', '全βタイプ'
        ]
    
    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.encoder(node_features))
        x = F.relu(self.gcn1(x, adj))
        x = F.relu(self.gcn2(x, adj))
        graph_emb = x.mean(dim=0, keepdim=True)
        logits = self.classifier(graph_emb)
        return logits, x


class ProteinGraphBuilder:
    """
    タンパク質をグラフに変換
    
    【やさしい説明】
    タンパク質の配列を「地図」に変換します。
    
    各アミノ酸に「性格」を付けます：
    - 水が嫌い（疎水性）か好き（親水性）か
    - 電気を帯びているか
    - 大きいか小さいか
    
    そして「誰と近くにいるか」の地図を作ります。
    """
    
    AA_PROPS = {
        'A': [1.8, 0, 0.3, 0], 'C': [2.5, 0, 0.4, 0], 'D': [-3.5, -1, 0.5, 1],
        'E': [-3.5, -1, 0.6, 1], 'F': [2.8, 0, 0.8, 0], 'G': [-0.4, 0, 0.1, 0],
        'H': [-3.2, 0.5, 0.7, 1], 'I': [4.5, 0, 0.6, 0], 'K': [-3.9, 1, 0.7, 1],
        'L': [3.8, 0, 0.6, 0], 'M': [1.9, 0, 0.7, 0], 'N': [-3.5, 0, 0.5, 1],
        'P': [-1.6, 0, 0.4, 0], 'Q': [-3.5, 0, 0.6, 1], 'R': [-4.5, 1, 0.8, 1],
        'S': [-0.8, 0, 0.3, 1], 'T': [-0.7, 0, 0.4, 1], 'V': [4.2, 0, 0.5, 0],
        'W': [-0.9, 0, 1.0, 0], 'Y': [-1.3, 0, 0.9, 1],
    }
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
    
    def build_graph(self, sequence: str, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """配列からグラフを構築"""
        n = len(sequence)
        
        # ノード特徴量
        features = []
        for aa in sequence.upper():
            props = self.AA_PROPS.get(aa, [0, 0, 0, 0])
            one_hot = [1.0 if AMINO_ACIDS[i] == aa else 0.0 for i in range(20)]
            feat = props + one_hot
            feat += [0.0] * (self.feature_dim - len(feat))
            features.append(feat[:self.feature_dim])
        
        # 隣接行列（k近傍）
        adj = torch.zeros(n, n)
        for i in range(n):
            for j in range(max(0, i-k), min(n, i+k+1)):
                adj[i, j] = 1.0
        
        # 正規化
        degree = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        adj = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
        
        return torch.tensor(features, dtype=torch.float32), adj


class ProteinStructureAnalyzer:
    """
    タンパク質構造分析器
    
    【やさしい説明】
    これは「タンパク質の形を予測する先生」です。
    
    配列を見て、「この子はらせん状（αヘリックス）になりそう」とか
    「平らなシート状（βシート）になりそう」と教えてくれます。
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.graph_builder = ProteinGraphBuilder()
        self.gcn = ProteinGCN().to(self.device)
    
    def analyze(self, sequence: str) -> Dict[str, float]:
        """構造を予測"""
        features, adj = self.graph_builder.build_graph(sequence)
        features, adj = features.to(self.device), adj.to(self.device)
        
        self.gcn.eval()
        with torch.no_grad():
            logits, _ = self.gcn(features, adj)
            probs = F.softmax(logits, dim=-1).squeeze()
        
        return {name: prob.item() for name, prob in zip(self.gcn.structure_names, probs)}


def demo():
    """デモンストレーション"""
    print("=" * 60)
    print("🧬 ProteinAI - タンパク質構造予測 (GCN)")
    print("=" * 60)
    print()
    print("【このプログラムができること】")
    print("タンパク質がどんな「形」に折りたたまれるかを予測します。")
    print()
    
    analyzer = ProteinStructureAnalyzer()
    
    alpha_seq = "AEEAAKAFIEAIKALK"  # αヘリックスになりやすい配列
    print(f"入力配列: {alpha_seq}")
    print("（ヘリックスを形成しやすい配列）")
    print()
    
    print("【予測される構造】")
    results = analyzer.analyze(alpha_seq)
    for struct, prob in sorted(results.items(), key=lambda x: -x[1])[:5]:
        bar = "█" * int(prob * 20)
        print(f"  {struct}: {prob:.1%} {bar}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    demo()
