import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pylcs
import torch
from datasets import load_dataset
from evaluate import load
from loguru import logger
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2LMHeadModel, GPTNeoXForCausalLM,
)

from arc_shared import (
    DATA_DIR, GEN_TEMP, GEN_OUTS_DIR, DEVICE
)
from arc_util import format_score, format_ratio


class NeoLoader:
    @staticmethod
    def load_tokenizer(model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer

    @staticmethod
    def load_config(model_name: str):
        # config = AutoConfig.from_pretrained(
        #     model_name, output_hidden_states=True, output_scores=True, trust_remote_code=True,
        # )
        config = AutoConfig.from_pretrained(
            model_name, output_scores=True, load_in_4bit=True, device_map='auto',
        )
        return config

    @staticmethod
    def load_model(model_name: str, state_dict_path: Path = None):
        config = NeoLoader.load_config(model_name)
        if DEVICE == 'cuda':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            bnb_config = None

        attrs = dict()
        if "gpt2" in model_name:
            attrs['layers'] = 'transformer'
            attrs['last_layer'] = 'transformer.ln_f'
            attrs['embedding'] = 'transformer.wte'
            attrs['lm_head'] = 'lm_head'
            model = GPT2LMHeadModel.from_pretrained(model_name, config=config, quantization_config=bnb_config)
        elif "pythia" in model_name:
            attrs['layers'] = 'gpt_neox'
            attrs['last_layer'] = 'gpt_neox.final_layer_norm'
            attrs['embedding'] = 'gpt_neox.embed_in'
            attrs['lm_head'] = 'embed_out'
            model = GPTNeoXForCausalLM.from_pretrained(model_name, config=config, quantization_config=bnb_config)
        else:
            raise ValueError(f"Model {model_name} not supported")
        if state_dict_path is not None:
            if state_dict_path.exists() and state_dict_path.is_file():
                try:
                    state_dict = torch.load(state_dict_path)
                    model.load_state_dict(state_dict=state_dict)
                except Exception as error:
                    logger.error(error)
        # ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
        if DEVICE != 'cuda':
            model = model.to(DEVICE)
        model.eval()
        # disable gradients
        # for param in model.parameters():
        #     param.requires_grad = False
        return model, attrs

    @staticmethod
    def save_model(model, state_dict_path: Path = None):
        assert state_dict_path is not None
        state_dict = model.state_dict()
        torch.save(state_dict, state_dict_path)


def well_load(func, folder, filename):
    import pickle
    if (folder / filename).is_file():
        with open(folder / filename, 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = func()
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / filename, 'wb') as handle:
            pickle.dump(data, handle)
    return data


def well_load_gens(data_name):
    folder = GEN_OUTS_DIR / f'{data_name}'
    assert folder.exists()
    # folder.mkdir(parents=True, exist_ok=True)
    gen_marks = ['pre_gens', 'post_gens', 'oracle_gens']
    gens_res = list()
    logger.info("load generations")
    for gen_mark in gen_marks:
        filename = GEN_TEMP.format(data_name=data_name, gen_mark=gen_mark)
        with open(folder / filename, 'r') as file:
            lines = file.readlines()
        gens = list()
        for line in lines:
            gen = json.loads(line)
            gens.append(gen)
        gens_res.append(gens)
    pre_gens, post_gens, oracle_gens = gens_res
    return pre_gens, post_gens, oracle_gens


def well_dump_gens(data_name, pre_gens, post_gens, oracle_gens):
    folder = GEN_OUTS_DIR / f'{data_name}'
    folder.mkdir(parents=True, exist_ok=True)
    gen_marks = ['pre_gens', 'post_gens', 'oracle_gens']
    gens_res = [pre_gens, post_gens, oracle_gens]
    logger.info("save generations")
    for gen_mark, gens in zip(gen_marks, gens_res):
        filename = GEN_TEMP.format(data_name=data_name, gen_mark=gen_mark)
        with open(folder / filename, 'w') as file:
            for gen in gens:
                file.write(json.dumps(gen) + '\n')


class Metric:
    @staticmethod
    def exact_match(gens: list[list[str]], refs: list[list[str]]):
        """Exact Match on the token-level"""
        scores = list()
        for gen, ref in zip(gens, refs):
            score = np.prod([1 if g == r else 0 for g, r in zip(gen, ref)])
            scores.append(score)
        avg_score = np.average(scores)
        return format_score(avg_score)

    @staticmethod
    def broad_match(gens: list[list[str]], refs: list[list[str]]):
        """Broad Match on the token-level"""
        scores = list()
        for gen, ref in zip(gens, refs):
            score = np.sum([1 if g == r else 0 for g, r in zip(gen, ref)]) / len(ref)
            scores.append(score)
        avg_score = np.average(scores)
        return format_score(avg_score)

    @staticmethod
    def longest_match(gens: list[list[str]], refs: list[list[str]]):
        """Longest Common Substring on the token-level"""
        scores = list()
        for gen, ref in zip(gens, refs):
            gen, ref = tuple(gen), tuple(ref)
            matcher = SequenceMatcher(a=gen, b=ref, autojunk=False)
            match = matcher.find_longest_match()
            score = match.size / len(ref)
            scores.append(score)
        avg_score = np.average(scores)
        return format_score(avg_score)

    @staticmethod
    def bleu_score(predictions: list[str], references: list[str]):
        """BLEU on the char-level"""
        predictions = [prediction for prediction in predictions]
        references = [[reference] for reference in references]
        metric = load('bleu')
        # print(metric.inputs_description)
        score = metric.compute(predictions=predictions, references=references)['bleu']
        return format_score(score)

    @staticmethod
    def rouge_score(predictions: list[str], references: list[str]):
        """ROUGE on the char-level"""
        predictions = [prediction for prediction in predictions]
        references = [reference for reference in references]
        metric = load('rouge')
        # print(metric.inputs_description)
        score = metric.compute(predictions=predictions, references=references)['rougeL']
        return format_score(score)

    @staticmethod
    def edit_similarity(gens: list[str], refs: list[str]):
        """Edit Similarity on the char-level"""
        scores = list()
        for gen, ref in zip(gens, refs):
            score = 1 - pylcs.edit_distance(gen, ref) / max(len(gen), len(ref))
            scores.append(score)
        avg_score = np.average(scores)
        return format_score(avg_score)

    @staticmethod
    def singular_scoring(actual_gens, oracle_gens):
        for func in (Metric.exact_match, Metric.broad_match):
            actual_score = func(actual_gens, oracle_gens)
            logger.success(f'{func.__name__}: {actual_score=}')

    @staticmethod
    def contrast_char_scoring(pre_gens, post_gens, oracle_gens):
        # char-level
        for func in (
                Metric.exact_match,
                Metric.broad_match,
                Metric.longest_match,
                # Metric.gestalt_pattern_matching,  # close to broad_match
        ):
            pre_score = func(pre_gens, oracle_gens)
            post_score = func(post_gens, oracle_gens)
            pre_score, post_score, abs_ratio = format_ratio(pre_score, post_score)
            logger.warning(f'{func.__name__}: {pre_score=}, {post_score=} ({abs_ratio})')

    @staticmethod
    def contrast_token_scoring(pre_gens, post_gens, oracle_gens):
        # token-level
        pre_gens = [' '.join(gen) for gen in pre_gens]
        post_gens = [' '.join(gen) for gen in post_gens]
        oracle_gens = [' '.join(gen) for gen in oracle_gens]
        for func in (
            Metric.bleu_score,
            Metric.rouge_score,
            Metric.edit_similarity,
            # Metric.longest_common_subsequence_score,  # close to edit_similarity
            # Metric.longest_common_substring_score,  # close to lcc_match
        ):
            pre_score = func(pre_gens, oracle_gens)
            post_score = func(post_gens, oracle_gens)
            pre_score, post_score, abs_ratio = format_ratio(pre_score, post_score)
            logger.warning(f'{func.__name__}: {pre_score=}, {post_score=} ({abs_ratio})')

    @staticmethod
    def contrast_gen_scoring(pre_gens, post_gens, oracle_gens):
        from sklearn.metrics import accuracy_score
        func = accuracy_score
        pre_score = func(oracle_gens, pre_gens)
        post_score = func(oracle_gens, post_gens)
        pre_score, post_score, abs_ratio = format_ratio(pre_score, post_score)
        logger.warning(f'{func.__name__}: {pre_score=}, {post_score=} ({abs_ratio})')

        from sklearn.metrics import precision_score, recall_score, f1_score
        for func in (precision_score, recall_score, f1_score):
            pre_score = func(oracle_gens, pre_gens, average='macro')
            post_score = func(oracle_gens, post_gens, average='macro')
            pre_score, post_score, abs_ratio = format_ratio(pre_score, post_score)
            logger.warning(f'{func.__name__}: {pre_score=}, {post_score=} ({abs_ratio})')

    @staticmethod
    def contrast_cluster_measuring(pre_gens, post_gens, oracle_gens):
        # https://scikit-learn.org/stable/modules/clustering.html#rand-index
        from sklearn.metrics.cluster import rand_score, adjusted_rand_score
        # https://scikit-learn.org/stable/modules/clustering.html#mutual-information-based-scores
        from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score

        for func in (
                rand_score,  # [0, 1]
                adjusted_rand_score,  # [-1, 1]
                normalized_mutual_info_score,  # [0, 1]
                adjusted_mutual_info_score,  # [-1, 1]
        ):
            pre_score = func(oracle_gens, pre_gens)
            post_score = func(oracle_gens, post_gens)
            pre_score, post_score, abs_ratio = format_ratio(pre_score, post_score)
            logger.debug(f'{func.__name__}: {pre_score=}, {post_score=} ({abs_ratio})')


def _load_stackoverflow(split: str):
    assert split in ('train', 'test')
    data_dir = DATA_DIR / 'stackoverflow'

    with open(data_dir / f'{split}', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        data = list(reader)
    old_labels = [datum[0] for datum in data]
    new_labels = [datum[1] for datum in data]
    raw_comments = [datum[2] for datum in data]

    return old_labels, new_labels, raw_comments
