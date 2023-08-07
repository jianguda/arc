import csv
import math
import pickle
import time
from collections import defaultdict
from queue import Queue, PriorityQueue
from sentence_transformers import util
from sklearn import metrics
from statistics import fmean

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import transformers
from loguru import logger
from tqdm import tqdm

from arc_shared import CFG_PROMPT_STYLE, REPORT_OUTS_DIR, DEVICE


def get_attr(mod: nn.Module, attrs: str):
    # from operator import attrgetter
    # embed_retriever = attrgetter("transformer.wte.weight")
    # self.model_embeddings = embed_retriever(self.model)
    for attr in attrs.split("."):
        mod = getattr(mod, attr)
    return mod


def set_attr(mod: nn.Module, attrs: str, new_mod: nn.Module):
    for attr in attrs.split(".")[:-1]:
        mod = getattr(mod, attr)
    setattr(mod, attrs.split(".")[-1], new_mod)


def evaluate(refs, gens):
    nmi = metrics.normalized_mutual_info_score(refs, gens)
    ari = metrics.adjusted_rand_score(refs, gens)
    fm = metrics.fowlkes_mallows_score(refs, gens)
    return nmi, ari, fm


###
def stabilize(reproducibility=True, seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def stylize(demo_pairs, query, data_name=None):
    probing_context = str()
    input_indicator = 'Query' if data_name in ['conala', 'django', 'spoc'] else 'Query'
    output_indicator = 'Code' if data_name in ['conala', 'django', 'spoc'] else 'API'
    for (demo_desc, demo_api) in demo_pairs:
        if CFG_PROMPT_STYLE == 'colon':
            probing_demo = f'{demo_desc}:{demo_api}\n'
        elif CFG_PROMPT_STYLE == 'lines':
            probing_demo = f'{input_indicator}:{demo_desc}\n{output_indicator}:{demo_api}\n\n'
        else:
            raise NotImplementedError
        probing_context += probing_demo

    if CFG_PROMPT_STYLE == 'colon':
        probing_prefix = f'{query}:'
    elif CFG_PROMPT_STYLE == 'lines':
        probing_prefix = f'{input_indicator}:{query}\n{output_indicator}:'
    else:
        raise NotImplementedError

    return probing_context, probing_prefix


def print_info(model):
    print(model)
    for name, parameter in model.base_model.named_parameters():
        print(name, parameter.size())


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        results = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_time = format_score(end - start)
        logger.debug(f"{func.__name__} takes {elapsed_time} seconds")
        return elapsed_time, results

    return wrapper


def format_score(datum):
    return round(datum, 3)


def format_ratio(pre_score, post_score):
    sign_prefix = ('+' if post_score >= pre_score else '')
    abs_ratio = sign_prefix + f'{format_score((post_score - pre_score) * 100)}%'
    # rel_ratio = sign_prefix + f'{format_score((post_score / pre_score - 1.) * 100)}%'
    # return abs_ratio, rel_ratio
    pre_score = format_score(pre_score)
    post_score = format_score(post_score)
    return pre_score, post_score, abs_ratio


def inspect(epoch, identifier, train_embeds, train_labels):
    # TODO theory analysis
    # the upper/lower bounds of knn-prompting
    # the expected performance of ARC

    # train_logits = self.embeds_to_logits(train_embeds)
    # anchors = train_logits
    anchors = train_embeds.squeeze(1)

    # logger.debug(f'{train_embeds.shape=}')
    # logger.debug(f'{len(train_labels)=}')
    # logger.debug(f'{train_embeds[0]=}')
    # logger.debug(f'{train_labels[:10]=}')
    token_families = defaultdict(list)
    for train_idx, train_label in enumerate(train_labels):
        token_families[train_label].append(train_idx)

    # TODO this is by COS
    def get_knn_idx(queries, anchors, k=1):
        # return idx of k nearest neighbours of the query
        # query: logits
        # neighbour_idx: [idx_1, ...]

        size = len(queries)
        queries = torch.stack(queries, dim=0).to(DEVICE)
        queries = queries.squeeze(1).to(DEVICE)
        anchors = anchors.squeeze(1).to(DEVICE)
        # logger.warning(f'{queries.shape=}')
        # logger.warning(f'{anchors.shape=}')
        simi_matrix = util.cos_sim(queries, anchors)
        # logger.debug(f'{simi_matrix.shape=}')
        simi_matrix = simi_matrix.detach().cpu().numpy()

        neighbors = list()
        for idx in range(size):
            similarities = simi_matrix[idx]
            neighbor_ids = np.argpartition(similarities, -k)[-k:]
            neighbors.append(neighbor_ids)

        return neighbors

    survey = dict()
    ks = [2 ** exp for exp in range(11)]
    assert 1 in ks
    # TODO optimize the algo...
    for label, token_family in token_families.items():
        # logger.debug(f'{train_embeds.shape=}')
        # logger.debug(f'{anchors.shape=}')

        # by COS
        member_rates = defaultdict(list)
        queries = list()
        for member_idx in tqdm(token_family):
            query = train_embeds[member_idx].clone().detach()
            # query = torch.tensor(train_logits[token_member])
            queries.append(query)

        top_k = min(ks[-1], len(token_family))
        all_neighbors = get_knn_idx(queries, anchors, k=top_k)
        for neighbors in all_neighbors:
            # logger.warning(f'{neighbors=}')
            for k in ks:
                top_neighbors = neighbors[:k]
                member_rate = fmean([1 if train_labels[idx] == label else 0 for idx in top_neighbors])
                member_rates[k].append(member_rate)

        # ...
        avg_member_rates = [format_score(fmean(member_rates[k])) for k in ks]
        survey[label] = avg_member_rates

    def measure(survey, power):
        measures = [rates[power] for rates in survey.values()]
        indicator = format_score(fmean(measures))
        return indicator

    """
    定义生成同token的prompts为同族prompts，假设第i个token族的数据有t_i + 1条，统计每条数据的k个最近邻居里面同族占比是多少
    其中，t_i ∈ N，k ∈ [1, t_i]，统计结果的数量是t_i * (t_i + 1)

    理想情况：族间距离总是大于族内距离，就是说，对每个样本，它是k个邻居都是同族的
    次优情况：调整参数k，使得对每个样本，它的k个邻居总是有超过半数是同族的
    对特定的token而言，只要能满足次优情况记忆增强就对模型生成有积极作用

    ## 优化指标（用来论证在取不同的k值时，根据不同的k取值对最终效果的影响，我们的方法比kp更好）
    1. 计算效果预期
    对符合t_i<=k情形的token族，计算效果预期（直接计算占比列表中的所有数值的均值）
    2. 计算效果上限和效果下限
    对符合t_i>k情形的token族，计算效果上限或者下限（考虑发生子采样的最好或者最坏情况）
    效果上限：对该占比列表排序，计算每个占比列表中数值最大的k个元素的求和
    效果下限：对该占比列表排序，计算每个占比列表中数值最小的k个元素的求和
    """
    folder = REPORT_OUTS_DIR
    filename = f'{identifier}.epoch.csv'
    avg_indicators = [measure(survey, idx) for idx in range(len(ks))]
    for k, avg_member_rate in zip(ks, avg_indicators):
        logger.debug(f'AVG {k=}, {avg_member_rate=}')

    header = ['epoch'] + [f'{k=}' for k in ks]
    avg_raw = [str(epoch)] + avg_indicators
    folder.mkdir(parents=True, exist_ok=True)
    if epoch == 0:
        with open(folder / filename, 'w', encoding='utf-8') as file:
            writer = csv.DictWriter(file, header)
            writer.writeheader()
            writer.writerow(dict(zip(header, avg_raw)))
    else:
        with open(folder / filename, 'a', encoding='utf-8') as file:
            writer = csv.DictWriter(file, header)
            writer.writerow(dict(zip(header, avg_raw)))
