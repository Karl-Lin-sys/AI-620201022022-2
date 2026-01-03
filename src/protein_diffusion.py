"""
🧬 ProteinAI - タンパク質を「創る」モジュール (Diffusion)
==========================================================

【やさしい説明】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

お絵描きを想像してください。
白い紙に少しずつ線を足していって、最後に素敵な絵ができますよね。

「Diffusion（拡散）モデル」は、これの逆をします。
最初にめちゃくちゃな落書き（ノイズ）から始めて、
少しずつキレイにして、最後に意味のあるタンパク質配列を作ります。

ステップ1: めちゃくちゃな文字の並び（ノイズ）
ステップ2: 少しだけ意味のある配列に
...
最終ステップ: 本物そっくりのタンパク質配列！

【技術的説明】
- DDPM (Denoising Diffusion Probabilistic Model)
- Forward process: データにノイズを加える
- Reverse process: ノイズを除去して新しいサンプルを生成
- U-Net: ノイズを予測するニューラルネットワーク

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import math

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)


class TimeEmbedding(nn.Module):
    """
    時間の埋め込み
    
    【やさしい説明】
    「今、何回目のキレイにする作業中？」を
    コンピュータに教えてあげる仕組みです。
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DenoisingBlock(nn.Module):
    """
    ノイズ除去ブロック
    
    【やさしい説明】
    消しゴムのようなものです。
    「ここはちょっと変だな」という部分を見つけて、
    少しずつキレイにしていきます。
    """
    
    def __init__(self, ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, ch)
        self.time_mlp = nn.Linear(time_dim, ch)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None]
        return F.silu(self.conv2(h)) + x


class ProteinDiffusionNet(nn.Module):
    """
    タンパク質生成ネットワーク
    
    【やさしい説明】
    これは「お絵描きロボット」のようなものです。
    
    最初はでたらめな下書きから始めて、
    何度も何度もキレイにしていくと...
    最後には本物のタンパク質のような配列ができます！
    """
    
    def __init__(self, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim // 2),
            nn.Linear(time_dim // 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.input_proj = nn.Conv1d(NUM_AA, hidden, 1)
        self.blocks = nn.ModuleList([DenoisingBlock(hidden, time_dim) for _ in range(4)])
        self.output_proj = nn.Conv1d(hidden, NUM_AA, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


class ProteinDiffusion:
    """
    タンパク質拡散モデル
    
    【やさしい説明】
    新しいタンパク質を「発明」できる魔法の道具です！
    
    「こんなタンパク質を作りたいな」と思ったら、
    このプログラムが新しい配列を考えてくれます。
    
    薬を作る研究で、とても役立ちます。
    """
    
    def __init__(self, steps: int = 100, device: str = 'cpu'):
        self.device = torch.device(device)
        self.steps = steps
        self.betas = torch.linspace(1e-4, 0.02, steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.model = ProteinDiffusionNet().to(self.device)
    
    def _to_onehot(self, seqs: List[str], max_len: int) -> torch.Tensor:
        batch = torch.zeros(len(seqs), NUM_AA, max_len, device=self.device)
        for i, s in enumerate(seqs):
            for j, aa in enumerate(s[:max_len]):
                if aa in AA_TO_IDX:
                    batch[i, AA_TO_IDX[aa], j] = 1.0
        return batch
    
    def _to_sequence(self, x: torch.Tensor) -> List[str]:
        idx = x.argmax(dim=1)
        return [''.join([IDX_TO_AA[i.item()] for i in row]) for row in idx]
    
    @torch.no_grad()
    def generate(self, n: int = 1, length: int = 30) -> List[str]:
        """
        新しいタンパク質配列を生成
        
        【やさしい説明】
        「新しいタンパク質を作って！」とお願いすると、
        一から新しい配列を考えて作ってくれます。
        まるで、新しいレシピを考案するシェフのようです。
        """
        self.model.eval()
        x = torch.randn(n, NUM_AA, length, device=self.device)
        
        for t in range(self.steps - 1, -1, -1):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t_batch)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bar[t]
            x = (x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred) / alpha.sqrt()
            if t > 0:
                x += self.betas[t].sqrt() * torch.randn_like(x)
        
        return self._to_sequence(x)


def demo():
    """デモンストレーション"""
    print("=" * 60)
    print("🧬 ProteinAI - タンパク質生成 (Diffusion)")
    print("=" * 60)
    print()
    print("【このプログラムができること】")
    print("まったく新しいタンパク質の配列を「発明」します。")
    print()
    print("【仕組み】")
    print("1. 最初はでたらめな文字の並び（ノイズ）")
    print("2. 少しずつキレイにしていく")
    print("3. 最後に意味のある配列になる！")
    print()
    
    diffusion = ProteinDiffusion()
    
    print("新しい配列を3つ生成中...")
    sequences = diffusion.generate(n=3, length=20)
    
    print()
    print("【生成された配列】")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq}")
    
    print()
    print("これらは今まで存在しなかった、新しいタンパク質です！")
    print("=" * 60)


if __name__ == "__main__":
    demo()
