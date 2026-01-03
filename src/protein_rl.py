"""
🧬 ProteinAI - タンパク質を「改良」するモジュール (RL)
======================================================

【やさしい説明】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ゲームで「もっと強いキャラを育てたい！」と思ったことはありますか？
敵を倒すと経験値がもらえて、だんだん強くなりますよね。

「強化学習（RL = Reinforcement Learning）」も同じ考え方です。

タンパク質を「もっと良くしたい！」と思ったら：
1. ちょっと変えてみる（変異を入れる）
2. 良くなったら「ご褒美」、悪くなったら「やり直し」
3. 何度も繰り返して、最高のタンパク質にする！

薬の開発では「病気に効く」「副作用が少ない」タンパク質を
育てていくのに使います。

【技術的説明】
- Policy Gradient: 行動方策の最適化
- Reward Function: 目標特性への報酬設計
- Value Function: 現在の状態の価値推定

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
NUM_AA = len(AMINO_ACIDS)


class PolicyNetwork(nn.Module):
    """
    方策ネットワーク
    
    【やさしい説明】
    これは「コーチ」のようなものです。
    「次はこのアミノ酸に変えてみよう！」とアドバイスしてくれます。
    
    何度も試行錯誤して、だんだん良いアドバイスができるようになります。
    """
    
    def __init__(self, seq_len: int = 20, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(NUM_AA, 32)
        self.lstm = nn.LSTM(32, hidden, batch_first=True, bidirectional=True)
        self.policy = nn.Linear(hidden * 2, NUM_AA)
        self.value = nn.Linear(hidden * 2, 1)
    
    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(seq)
        h, _ = self.lstm(x)
        logits = self.policy(h)
        value = self.value(h.mean(dim=1))
        return logits, value


class ProteinEnvironment:
    """
    タンパク質の「世界」
    
    【やさしい説明】
    ゲームの「ステージ」のようなものです。
    
    タンパク質を変えたとき、「良くなった！+10点」「悪くなった...-5点」
    というスコアを計算してくれます。
    
    目標：
    - 水に溶けやすくしたい（親水性を上げる）
    - 電気的に中性にしたい（電荷をゼロに）
    """
    
    AA_PROPS = {
        'A': {'hydro': 1.8, 'charge': 0}, 'C': {'hydro': 2.5, 'charge': 0},
        'D': {'hydro': -3.5, 'charge': -1}, 'E': {'hydro': -3.5, 'charge': -1},
        'F': {'hydro': 2.8, 'charge': 0}, 'G': {'hydro': -0.4, 'charge': 0},
        'H': {'hydro': -3.2, 'charge': 0.5}, 'I': {'hydro': 4.5, 'charge': 0},
        'K': {'hydro': -3.9, 'charge': 1}, 'L': {'hydro': 3.8, 'charge': 0},
        'M': {'hydro': 1.9, 'charge': 0}, 'N': {'hydro': -3.5, 'charge': 0},
        'P': {'hydro': -1.6, 'charge': 0}, 'Q': {'hydro': -3.5, 'charge': 0},
        'R': {'hydro': -4.5, 'charge': 1}, 'S': {'hydro': -0.8, 'charge': 0},
        'T': {'hydro': -0.7, 'charge': 0}, 'V': {'hydro': 4.2, 'charge': 0},
        'W': {'hydro': -0.9, 'charge': 0}, 'Y': {'hydro': -1.3, 'charge': 0},
    }
    
    def __init__(self, target: Dict[str, float]):
        self.target = target
    
    def get_properties(self, seq: str) -> Dict[str, float]:
        """配列の性質を計算"""
        if not seq:
            return {'hydro': 0, 'charge': 0}
        hydro = sum(self.AA_PROPS.get(aa, {}).get('hydro', 0) for aa in seq) / len(seq)
        charge = sum(self.AA_PROPS.get(aa, {}).get('charge', 0) for aa in seq)
        return {'hydro': hydro, 'charge': charge}
    
    def get_reward(self, seq: str) -> float:
        """スコア（ご褒美）を計算"""
        props = self.get_properties(seq)
        score = 0
        for k, v in self.target.items():
            if k in props:
                score -= abs(props[k] - v)  # 目標に近いほど高スコア
        return score


class ProteinRL:
    """
    タンパク質強化学習
    
    【やさしい説明】
    これは「タンパク質育成ゲーム」のようなものです！
    
    1. 「この性質を持つタンパク質が欲しい！」と目標を設定
    2. AIが色々な配列を試してみる
    3. 良いものができたら「やったね！」、ダメなら「もう一回」
    4. 何度も繰り返して、最高のタンパク質を見つける
    
    薬の開発で「副作用が少なくて、よく効く」タンパク質を
    見つけるのに役立ちます。
    """
    
    def __init__(self, seq_len: int = 20, device: str = 'cpu'):
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.policy = PolicyNetwork(seq_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.env = ProteinEnvironment({'hydro': 0, 'charge': 0})
    
    def _idx_to_seq(self, idx: torch.Tensor) -> str:
        return ''.join([AMINO_ACIDS[i] for i in idx.tolist()])
    
    def _sample_action(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def optimize(self, target: Dict[str, float], steps: int = 30) -> List[Tuple[str, float]]:
        """
        目標に向けて最適化
        
        【やさしい説明】
        「この性質のタンパク質が欲しい！」と言うと、
        AIが一生懸命探して、最高の配列を見つけてくれます。
        """
        self.env.target = target
        results = []
        
        for _ in range(steps):
            self.policy.train()
            init = torch.randint(0, NUM_AA, (1, self.seq_len), device=self.device)
            logits, value = self.policy(init)
            action, log_prob = self._sample_action(logits)
            seq = self._idx_to_seq(action[0])
            reward = self.env.get_reward(seq)
            
            advantage = reward - value.item()
            loss = -(log_prob.sum() * advantage) + F.mse_loss(value, torch.tensor([[reward]], device=self.device))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            results.append((seq, reward))
        
        return results


def demo():
    """デモンストレーション"""
    print("=" * 60)
    print("🧬 ProteinAI - タンパク質最適化 (強化学習)")
    print("=" * 60)
    print()
    print("【このプログラムができること】")
    print("「こんな性質のタンパク質が欲しい！」という目標に向けて、")
    print("最適な配列を探し出します。")
    print()
    print("【目標】")
    print("  - 水に溶けやすい（疎水性 → -2）")
    print("  - 電気的に中性（電荷 → 0）")
    print()
    
    rl = ProteinRL()
    target = {'hydro': -2.0, 'charge': 0}
    
    print("最適化中...")
    results = rl.optimize(target, steps=20)
    
    best = max(results, key=lambda x: x[1])
    props = rl.env.get_properties(best[0])
    
    print()
    print("【結果】")
    print(f"  最良の配列: {best[0]}")
    print(f"  スコア: {best[1]:.2f}")
    print(f"  疎水性: {props['hydro']:.2f}（目標: -2.0）")
    print(f"  電荷: {props['charge']:.1f}（目標: 0）")
    print()
    print("=" * 60)


if __name__ == "__main__":
    demo()
