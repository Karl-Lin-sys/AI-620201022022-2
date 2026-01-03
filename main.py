"""
🧬 ProteinAI Research Platform - メインプログラム
"""

from src.protein_llm import ProteinLLM, demo as llm_demo
from src.protein_gcn import ProteinStructureAnalyzer, demo as gcn_demo
from src.protein_diffusion import ProteinDiffusion, demo as diffusion_demo
from src.protein_rl import ProteinRL, demo as rl_demo


def run_all_demos():
    """すべてのデモを実行"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " 🧬 ProteinAI Research Platform - 全デモ実行 ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    print("[1/4] LLM - タンパク質の「ことば」を理解")
    print("-" * 60)
    llm_demo()
    print()
    
    print("[2/4] GCN - タンパク質の「カタチ」を予測")
    print("-" * 60)
    gcn_demo()
    print()
    
    print("[3/4] Diffusion - 新しいタンパク質を「創る」")
    print("-" * 60)
    diffusion_demo()
    print()
    
    print("[4/4] RL - タンパク質を「育てる」")
    print("-" * 60)
    rl_demo()
    print()
    
    print("╔" + "═" * 58 + "╗")
    print("║" + " ✅ 全デモ完了！ ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    run_all_demos()
