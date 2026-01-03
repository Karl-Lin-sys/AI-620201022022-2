"""
🧬 ProteinAI - タンパク質のことば予測モジュール
==============================================

【やさしい説明】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

タンパク質は「アミノ酸」という小さな部品が鎖のようにつながったものです。
ちょうど、ひらがな（あ、い、う...）を並べて言葉を作るように、
アミノ酸（A, C, D, E...の20種類）を並べてタンパク質ができます。

このプログラムは「Transformer」という賢い仕組みを使って、
タンパク質の「言葉」を読み解き、その働きを予測します。

例えば「MVLSPADKTN...」という配列を見て、
「これはヘモグロビンかな？酸素を運ぶ仕事をしそうだね」と予測するのです。

【技術的説明】
- Transformer: 言語モデル（GPTのような技術）をタンパク質配列に応用
- Self-Attention: 配列中の遠く離れた要素同士の関係性を学習
- Few-shot Learning: 少数の例から新しいタンパク質の機能を推測

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

# アミノ酸のアルファベット（タンパク質の「ひらがな」のようなもの）
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'  # 20種類のアミノ酸
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS) + 2  # パディングと未知文字用


class PositionalEncoding(nn.Module):
    """
    位置エンコーディング
    
    【やさしい説明】
    文章を読むとき「最初の文字」「２番目の文字」と順番が大事ですよね。
    コンピュータにも「これは何番目にある文字か」を教えてあげる仕組みです。
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]


class ProteinTransformer(nn.Module):
    """
    タンパク質言語モデル
    
    【やさしい説明】
    本を読む時、前後の文脈を見て意味を理解しますよね。
    このモデルも同じように、タンパク質配列の前後を見て
    「このタンパク質は何をするものか」を理解しようとします。
    
    「Self-Attention（自己注意）」という技術で、
    配列の中でどの部分が大切かを自分で見つけます。
    """
    
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self.embedding(src) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.norm(x)


class ProteinFunctionPredictor(nn.Module):
    """
    タンパク質機能予測器
    
    【やさしい説明】
    タンパク質には様々な「仕事」があります。
    - 酸素を運ぶ（ヘモグロビン）
    - 食べ物を消化する（消化酵素）
    - 病気と戦う（抗体）
    
    このモデルは、配列を見て「この子はどんな仕事をするのかな？」と
    予測してくれる、タンパク質の「職業診断」のようなものです。
    """
    
    def __init__(self, num_functions: int = 20, d_model: int = 256):
        super().__init__()
        self.encoder = ProteinTransformer(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_functions)
        )
        self.function_names = [
            'DNA結合', 'RNA結合', 'ATP結合', '金属結合', '酵素',
            '輸送体', '受容体', 'キナーゼ', 'プロテアーゼ', '酸化還元',
            '転移酵素', '加水分解', '脱離酵素', '異性化酵素', '連結酵素',
            '構造', 'シグナル', '免疫', '細胞接着', '転写因子'
        ]
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(src)
        pooled = encoded.mean(dim=0)
        return self.classifier(pooled)


class ProteinLLM:
    """
    タンパク質LLMインターフェース
    
    【やさしい説明】
    これは、難しい機械学習の部分を隠して、
    簡単に使えるようにした「リモコン」のようなものです。
    
    使い方：
    1. タンパク質の配列（文字の並び）を入力
    2. 「この子は何をするタンパク質？」と質問
    3. 答えが返ってくる
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = ProteinFunctionPredictor().to(self.device)
    
    def tokenize(self, sequence: str) -> torch.Tensor:
        """配列を数字に変換（コンピュータが読める形に）"""
        indices = [AA_TO_IDX.get(aa, len(AMINO_ACIDS) + 1) for aa in sequence.upper()]
        return torch.tensor(indices, dtype=torch.long, device=self.device)
    
    def predict_function(self, sequence: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        タンパク質の機能を予測
        
        【やさしい説明】
        「MVLSPADKTN...」のような配列を渡すと、
        「これは多分○○という働きをするよ」と教えてくれます。
        """
        tokens = self.tokenize(sequence).unsqueeze(1)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tokens)
            probs = F.softmax(logits, dim=-1).squeeze()
        
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.model.function_names)))
        return [(self.model.function_names[i], p.item()) for p, i in zip(top_probs, top_indices)]
    
    def encode(self, sequence: str) -> np.ndarray:
        """配列を数値ベクトルに変換"""
        tokens = self.tokenize(sequence).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            encoded = self.model.encoder(tokens)
        return encoded.mean(dim=0).squeeze().cpu().numpy()


def demo():
    """デモンストレーション"""
    print("=" * 60)
    print("🧬 ProteinAI - タンパク質機能予測")
    print("=" * 60)
    print()
    print("【このプログラムができること】")
    print("タンパク質の「文字列」を読んで、その働きを予測します。")
    print()
    
    llm = ProteinLLM()
    
    # インスリン（血糖値を調整するホルモン）
    insulin = "GIVEQCCTSICSLYQLENYCN"
    print(f"入力配列: {insulin}")
    print(f"配列長: {len(insulin)}アミノ酸")
    print()
    
    print("【予測される機能】")
    for func, prob in llm.predict_function(insulin):
        bar = "█" * int(prob * 20)
        print(f"  {func}: {prob:.1%} {bar}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    demo()
