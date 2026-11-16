"""Reuse the checked finite benchmark with the new Medium shapes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_bert_controlled as benchmark
from src import bert_medium
benchmark.SPECS=bert_medium.SPECS
benchmark.make_model=bert_medium.make_model
if __name__=='__main__': benchmark.main()
