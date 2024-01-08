import argparse

from lightning.pytorch.utilities.seed import isolate_rng
from loguru import logger

from arc_cfg import Configure
from arc_pipeline import Pipeline
from arc_util import stabilize


def main():
    logger.info(f'{args=}')
    cfg = Configure(args)

    stabilize()
    with isolate_rng():
        pipeline = Pipeline(cfg)
        pipeline.optimize_trainer()


# TODO
#  experiments: datasets, models, tasks, metrics
#  writings:
#   1. https://zhuanlan.zhihu.com/p/492684696 (distribution => representation)
#   2. white-box, invertible flow-based model (network)
#   3. clustering, RAG
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='trec', type=str, help='[stackoverflow] sst2 trec ...')
    parser.add_argument('--model', default='gpt2', type=str, help='[pythia-xs] gpt2 ...')
    parser.add_argument('--exp', default='standard.approach', type=str, help='[standard.approach] ...')
    parser.add_argument('--epoch-num', default='500', type=int, help='[1] 100 ...')
    parser.add_argument('--batch-size', default='1024', type=int, help='[1024] 2048 512 ...')
    args = parser.parse_args()

    main()


"""
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=1, avg_member_rate=0.389
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=2, avg_member_rate=0.398
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=4, avg_member_rate=0.424
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=8, avg_member_rate=0.461
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=16, avg_member_rate=0.52
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=32, avg_member_rate=0.595
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=64, avg_member_rate=0.693
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=128, avg_member_rate=0.786
2024-01-02 14:27:34.458 | DEBUG    | arc_util:inspect:193 - AVG k=256, avg_member_rate=0.855
2024-01-02 14:27:34.459 | DEBUG    | arc_util:inspect:193 - AVG k=512, avg_member_rate=0.897
2024-01-02 14:27:34.459 | DEBUG    | arc_util:inspect:193 - AVG k=1024, avg_member_rate=0.909
Epoch 99: 100%|██████████| 15/15 [00:07<00:00,  1.90it/s, v_num=73]
2024-01-02 14:27:35.187 | WARNING  | arc_focus:optimize_trainer:711 - TRAIN ...
/Users/jguu0040/anaconda3/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:441: The 'predict_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
Predicting DataLoader 0: 100%|██████████| 15/15 [00:00<00:00, 21.87it/s]
2024-01-02 14:37:07.544 | SUCCESS  | arc_load:contrast_char_scoring:236 - accuracy: pre_score=0.0, post_score=0.955 (+95.5%)
2024-01-02 14:38:03.150 | SUCCESS  | arc_load:contrast_char_scoring:236 - accuracy: pre_score=0.064, post_score=0.957 (+89.3%)
2024-01-02 14:38:03.159 | WARNING  | arc_load:contrast_cluster_measuring:275 - rand_score: pre_score=0.6565083275275211, post_score=0.991490624968926 (+33.498%)
2024-01-02 14:38:03.164 | WARNING  | arc_load:contrast_cluster_measuring:275 - adjusted_rand_score: pre_score=0.0018756255365564244, post_score=0.9130277059697247 (+91.115%)
2024-01-02 14:38:03.171 | WARNING  | arc_load:contrast_cluster_measuring:275 - normalized_mutual_info_score: pre_score=0.01249825744063641, post_score=0.9111333082268998 (+89.864%)
2024-01-02 14:38:03.192 | WARNING  | arc_load:contrast_cluster_measuring:275 - adjusted_mutual_info_score: pre_score=0.009052488981283933, post_score=0.9107727924029136 (+90.172%)
2024-01-02 14:38:03.197 | WARNING  | arc_load:contrast_cluster_measuring:275 - fowlkes_mallows_score: pre_score=0.1319987070739307, post_score=0.9175191015218228 (+78.552%)
2024-01-02 14:38:32.074 | SUCCESS  | arc_load:contrast_char_scoring:236 - accuracy: pre_score=0.064, post_score=0.328 (+26.4%)
2024-01-02 14:38:32.082 | WARNING  | arc_load:contrast_cluster_measuring:275 - rand_score: pre_score=0.6565083275275211, post_score=0.884577274720599 (+22.807%)
2024-01-02 14:38:32.087 | WARNING  | arc_load:contrast_cluster_measuring:275 - adjusted_rand_score: pre_score=0.0018756255365564244, post_score=0.09396500053148028 (+9.209%)
2024-01-02 14:38:32.095 | WARNING  | arc_load:contrast_cluster_measuring:275 - normalized_mutual_info_score: pre_score=0.01249825744063641, post_score=0.23680823956915134 (+22.431%)
2024-01-02 14:38:32.113 | WARNING  | arc_load:contrast_cluster_measuring:275 - adjusted_mutual_info_score: pre_score=0.009052488981283933, post_score=0.23356103384822147 (+22.451%)
2024-01-02 14:38:32.118 | WARNING  | arc_load:contrast_cluster_measuring:275 - fowlkes_mallows_score: pre_score=0.1319987070739307, post_score=0.15669201278138734 (+2.469%)
2024-01-02 14:38:32.118 | WARNING  | arc_focus:optimize_trainer:718 - TEST ...
/Users/jguu0040/anaconda3/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:441: The 'predict_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
Predicting DataLoader 0: 100%|██████████| 5/5 [00:00<00:00, 25.99it/s]
2024-01-02 14:41:42.616 | SUCCESS  | arc_load:contrast_char_scoring:236 - accuracy: pre_score=0.0, post_score=0.818 (+81.8%)
2024-01-02 14:42:00.387 | SUCCESS  | arc_load:contrast_char_scoring:236 - accuracy: pre_score=0.036, post_score=0.824 (+78.8%)
2024-01-02 14:42:00.391 | WARNING  | arc_load:contrast_cluster_measuring:275 - rand_score: pre_score=0.6844322838037377, post_score=0.9669039828442628 (+28.247%)
2024-01-02 14:42:00.393 | WARNING  | arc_load:contrast_cluster_measuring:275 - adjusted_rand_score: pre_score=0.00576789411215253, post_score=0.7110680848770913 (+70.53%)
2024-01-02 14:42:00.397 | WARNING  | arc_load:contrast_cluster_measuring:275 - normalized_mutual_info_score: pre_score=0.018822696382046134, post_score=0.7132473043400183 (+69.442%)
2024-01-02 14:42:00.405 | WARNING  | arc_load:contrast_cluster_measuring:275 - adjusted_mutual_info_score: pre_score=0.010522453759354707, post_score=0.7094635849207471 (+69.894%)
2024-01-02 14:42:00.407 | WARNING  | arc_load:contrast_cluster_measuring:275 - fowlkes_mallows_score: pre_score=0.14273595993284263, post_score=0.7293335405820109 (+58.66%)
2024-01-02 14:42:09.563 | SUCCESS  | arc_load:contrast_char_scoring:236 - accuracy: pre_score=0.036, post_score=0.287 (+25.1%)
2024-01-02 14:42:09.565 | WARNING  | arc_load:contrast_cluster_measuring:275 - rand_score: pre_score=0.6844322838037377, post_score=0.8670269020665753 (+18.259%)
2024-01-02 14:42:09.568 | WARNING  | arc_load:contrast_cluster_measuring:275 - adjusted_rand_score: pre_score=0.00576789411215253, post_score=0.07341081944469563 (+6.764%)
2024-01-02 14:42:09.571 | WARNING  | arc_load:contrast_cluster_measuring:275 - normalized_mutual_info_score: pre_score=0.018822696382046134, post_score=0.23189504037141312 (+21.307%)
2024-01-02 14:42:09.577 | WARNING  | arc_load:contrast_cluster_measuring:275 - adjusted_mutual_info_score: pre_score=0.010522453759354707, post_score=0.22109504352606096 (+21.057%)
2024-01-02 14:42:09.579 | WARNING  | arc_load:contrast_cluster_measuring:275 - fowlkes_mallows_score: pre_score=0.14273595993284263, post_score=0.1453380066582448 (+0.26%)
"""
