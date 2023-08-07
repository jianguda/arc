import logging
import os
from pathlib import Path

import torch

logging.disable(logging.WARNING)
os.environ['HF_HOME'] = '../../hf'
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_VERBOSITY'] = "error"
# if os.environ["PYTHONHASHSEED"] != "0":
#     logger.info("Warning, please set environment variable PYTHONHASHSEED to 0 for determinism")

# features
CFG_MODE_DRYRUN = False
CFG_MODE_INSPECT = False
CFG_MODE_VERBOSE = False
CFG_MODE_BASELINE = False

dryrun_mark = '_.' if CFG_MODE_DRYRUN else ''
baseline_mark = '=.' if CFG_MODE_BASELINE else ''

DATA_DIR = Path.cwd().parent / 'data'
_OUTS_DIR = Path.cwd().parent / 'outs'
SIMI_OUTS_DIR = _OUTS_DIR / 'simi'
VOCAB_OUTS_DIR = _OUTS_DIR / 'vocab'
LOG_OUTS_DIR = _OUTS_DIR / (baseline_mark + 'log')
GEN_OUTS_DIR = _OUTS_DIR / (baseline_mark + 'gen')
FIG_OUTS_DIR = _OUTS_DIR / (baseline_mark + 'fig')
STATS_OUTS_DIR = _OUTS_DIR / (baseline_mark + 'stats')
# DSTORE_OUTS_DIR = _OUTS_DIR / 'dstore'
# TWIST_OUTS_DIR = _OUTS_DIR / 'twist'
CACHE_OUTS_DIR = _OUTS_DIR / 'cache'
REPORT_OUTS_DIR = _OUTS_DIR / 'report'
VISUAL_OUTS_DIR = _OUTS_DIR / 'visual'

MATRIX_TEMP = dryrun_mark + 'matrix.{data_name}.{train_num}.{test_num}.pkl'
NEIGHBOR_TEMP = dryrun_mark + 'neighbor.{data_name}.{train_num}.{test_num}.pkl'
VOCAB_TEMP = dryrun_mark + 'vocab.{data_name}.{model_name}.pkl'
LOG_TEMP = dryrun_mark + 'me.{data_name}.{model_name}.log'
GEN_TEMP = dryrun_mark + 'me.{data_name}.{gen_mark}.jsonl'

MODEL_REGISTRY = {
    'gpt2': 'gpt2',
    'gpt2-m': 'gpt2-medium',
    'gpt2-l': 'gpt2-large',
    'gpt2-xl': 'gpt2-xl',
    'pythia-xs': 'EleutherAI/pythia-70m-deduped',
    'pythia-s': 'EleutherAI/pythia-160m-deduped',
    'pythia-m': 'EleutherAI/pythia-410m-deduped',
    'pythia-l': 'EleutherAI/pythia-1b-deduped',
    'pythia-xl': 'EleutherAI/pythia-1.4b-deduped',
    'pythia-xxl': 'EleutherAI/pythia-2.8b-deduped',
}

# ========== EXPERIMENT CONFIGURATION ==========
CFG_PROMPT_STYLE = 'colon'  # 'colon': separated by ':'
# CFG_PROMPT_STYLE = 'lines'  # 'lines': separated by '\n'

CFG_ANCHOR_INIT = 'abs'  # 'abs': ...
# CFG_ANCHOR_INIT = 'rel'  # 'rel': ...

# dryrun settings
DRYRUN_SAMPLE_NUM = 100

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# logger.add(_OUTS_DIR / 'me.log')
# LOG_FORMAT = '{time: YYYY-MM-DD HH:mm:ss} | {level} | {message}'
# logger.add(LOG_FILE, format=LOG_FORMAT)
