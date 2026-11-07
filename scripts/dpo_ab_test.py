"""Blind DPO versus unchanged pre-DPO model on fresh prompts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_ab_test as reviewer
reviewer.DIRECTORY=reviewer.ROOT/'data/human_review/dpo_ab_v1'
reviewer.main()
