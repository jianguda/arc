import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from lightning import Trainer
from loguru import logger
from openTSNE.affinity import MultiscaleMixture
from sentence_transformers import util
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_learner import CoupleDataset, Learner
from arc_load import NeoLoader, Metric, _load_stackoverflow
from arc_shared import DEVICE, CACHE_OUTS_DIR, CFG_ANCHOR_INIT, VISUAL_OUTS_DIR, DRYRUN_SAMPLE_NUM, CFG_MODE_DRYRUN, \
    CFG_MODE_VERBOSE, CFG_MODE_INSPECT
from arc_util import stylize, format_score, get_attr, inspect
from utils.dataset import load_dataset, get_max_demo_shot
from utils.template import make_prompt


class Pipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.model, self.attrs = NeoLoader.load_model(cfg.MODEL_NAME)
        self.tokenizer = NeoLoader.load_tokenizer(cfg.MODEL_NAME)

        # dim_size
        if 'gpt2' in cfg.MODEL_NAME:
            dim_size = self.model.config.n_embd
        elif 'pythia' in cfg.MODEL_NAME:
            dim_size = self.model.config.hidden_size
        else:
            raise NotImplementedError

        # ...
        self.identifier = f'{self.cfg.DATA_CODE}.{self.cfg.MODEL_CODE}'

        # cluster_num
        self.train_embeds, self.train_labels, self.test_embeds, self.test_labels = self.load_corpus()
        cluster_num = max(max(self.train_labels), max(self.test_labels)) + 1
        if CFG_MODE_DRYRUN:
            dry_factor_train = 9
            dry_factor_test = 1
            self.train_embeds = self.train_embeds[:DRYRUN_SAMPLE_NUM * dry_factor_train]
            self.train_labels = self.train_labels[:DRYRUN_SAMPLE_NUM * dry_factor_train]
            self.test_embeds = self.test_embeds[:DRYRUN_SAMPLE_NUM * dry_factor_test]
            self.test_labels = self.test_labels[:DRYRUN_SAMPLE_NUM * dry_factor_test]

        # ...
        self.learner = Learner(
            model=self.model, attrs=self.attrs, dim_size=dim_size, cluster_num=cluster_num, identifier=self.identifier)

        # we freeze all parameters to save computation
        for name, parameter in self.model.named_parameters():
            # print(name, parameter.size())
            parameter.requires_grad = False

    def load_corpus(self):
        folder = CACHE_OUTS_DIR

        filename = f'{self.identifier}.pkl'

        # load data
        if (folder / filename).is_file():
            with open(folder / filename, 'rb') as handle:
                train_embeds, train_labels, test_embeds, test_labels = pickle.load(handle)
            return train_embeds, train_labels, test_embeds, test_labels

        if self.cfg.DATA_CODE == 'stackoverflow':
            _, train_labels, train_comments = _load_stackoverflow('train')
            _, test_labels, test_comments = _load_stackoverflow('test')
            # label2id
            train_labels = [int(label) - 1 for label in train_labels]
            test_labels = [int(label) - 1 for label in test_labels]
        else:
            # prepare dataset
            train_data, dev_data = load_dataset(dataset=self.cfg.DATA_CODE)
            n_demo_shot = get_max_demo_shot(dataset=self.cfg.DATA_CODE)

            # TODO shall we try with few-shot demos???
            # train
            train_comments = list()
            train_labels = list()
            label2id = train_data.label2id
            for ins in train_data.data:
                comment = make_prompt(ins, self.cfg.DATA_CODE, mode='inference')
                train_comments.append(comment)
                label = label2id[ins['label']]
                train_labels.append(label)

            train_data.subsamplebyshot(n_demo_shot)
            prompt_prefix = make_prompt(train_data, self.cfg.DATA_CODE, mode='train')

            # test
            test_comments = list()
            test_labels = list()
            label2id = dev_data.label2id
            for ins in dev_data.data:
                comment = prompt_prefix + make_prompt(ins, self.cfg.DATA_CODE, mode='inference')
                test_comments.append(comment)
                label = label2id[ins['label']]
                test_labels.append(label)

        # em (we do like this one since it is intuitive and cheap)
        train_embeds = self.compose_data(train_comments)
        test_embeds = self.compose_data(test_comments)
        # # contrastive learning (we donot like this one since it is too costly)
        # train_i_embeds, train_j_embeds = cluster.assign_corpus(train_embeds, train_labels)

        # dump data
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / filename, 'wb') as handle:
            data = (train_embeds, train_labels, test_embeds, test_labels)
            pickle.dump(data, handle)

        return train_embeds, train_labels, test_embeds, test_labels

    # def assign_corpus(self, embeds, labels):
    #     embeds_i = list()
    #     embeds_j = list()
    #     size = len(embeds)
    #     for idx_i, idx_j in list(itertools.combinations(range(size), 2)):
    #         if labels[idx_i] == labels[idx_j]:
    #             embeds_i.append(embeds[idx_i])
    #             embeds_j.append(embeds[idx_j])
    #     return embeds_i, embeds_j

    def datum2tensor(self, datum):
        token_terms = datum.split()
        token_labels = [label for term in token_terms for label in self.tokenizer.encode(term)]
        # correct some out-of-vocabulary labels (might be caused by changed vocabulary)
        # however, it won't affect the length of tokenized token_labels
        token_labels = [self.tokenizer.encode(' ')[0] if label > 50256 else label for label in token_labels]
        token_values = [self.tokenizer.decode(label) for label in token_labels]
        return token_labels, token_values

    def _get_embeddings(self, token_labels) -> torch.Tensor:
        # embeddings_matrix = attrgetter(self.attrs['embedding'])(self.model)
        embeddings_matrix = get_attr(self.model, self.attrs['embedding'])
        embeddings_matrix = embeddings_matrix.weight
        inputs_embeds = [embeddings_matrix[token_label] for token_label in token_labels]
        inputs_embeds = torch.stack(inputs_embeds).unsqueeze(0)
        return inputs_embeds

    def extract_hidden_embeds(self, token_labels):
        # feature hook & partials
        features = dict()
        last_layer_feature_str = 'last_layer'
        last_layer = get_attr(self.model, self.attrs['layers'])
        # last_layer = get_attr(self.model, self.attrs['last_layer'])

        def last_layer_feature_hook(m, i, o):
            # when hooking a layer (for last layer)
            features[last_layer_feature_str] = o[0].detach()
            # logger.critical(f'{o[0].shape=}')

        # hook the last layer
        handle = last_layer.register_forward_hook(last_layer_feature_hook)
        # output = self.model(inputs_embeds=inputs_)
        # prob = F.softmax(output.logits[:, -1, :], dim=-1)
        # pred = output.detach().cpu().numpy()
        self.model.eval()
        # if we need the dists, use this
        # inputs_embeds = self._get_embeddings(token_labels)
        # inputs_embeds = inputs_embeds.unsqueeze(0)
        # output = self.model(inputs_embeds=inputs_embeds)
        # dists = F.softmax(output.logits[:, -1, :], dim=-1)
        # else, we directly use this one
        _ = self.model.generate(token_labels, max_new_tokens=1)
        layer_hidden_embeds = features[last_layer_feature_str]
        # remove the hook
        handle.remove()
        return layer_hidden_embeds

    def compose_data(self, comments):
        # matrix = self.learner.build_virtual_matrix()

        embeds_list = list()
        for comment in tqdm(comments):
            # TODO use this case to do case study, for example:
            #  1. after refinement, the actual case become close to the virtual case ...
            #  2. after refinement, the virtual case become close to the label family ...
            #   ...
            # logits = self._predict(comment)
            # logger.critical(f'{logits.shape=}')
            # # in the virtual case
            # layer_embed = torch.matmul(logits, matrix)
            # # logger.critical(f'{layer_embed.shape=}')
            # assert torch.sum(layer_embed.isnan()).item() == 0
            # # logger.critical(f'{torch.sum(layer_embed.isnan())=}')

            # in the actual case
            token_labels = self.tokenizer(comment, return_tensors="pt").input_ids.to(DEVICE)
            layer_embeds = self.extract_hidden_embeds(token_labels)
            # (multiple embeddings to one embedding)
            #  1. for `repr`, using the mean one
            # layer_embed = torch.mean(layer_embeds, dim=-2)
            #  2. for `gen`, using the last one
            layer_embed = layer_embeds[..., -1, :]
            # logger.debug(f'{layer_embed.shape=}')
            # logger.debug(f'{layer_embed.dtype=}')

            embeds_list.append(layer_embed)
        return embeds_list

    def logits_to_output(self, logits, watch_labels=None):
        # pred = torch.argmax(logits).tolist()
        sorted_probs, sorted_labels = logits.sort(dim=-1, descending=True)
        argmax_probs = sorted_probs.cpu().numpy()[0].tolist()
        argmax_labels = sorted_labels.cpu().numpy()[0].tolist()

        # # filter sorted_indices using the domain vocabs
        # if self.allow_vocab_labels is not None:
        #     argmax_vocab_probs = list()
        #     argmax_vocab_labels = list()
        #     for argmax_prob, argmax_label in zip(argmax_probs, argmax_labels):
        #         if argmax_label in self.allow_vocab_labels:
        #             argmax_vocab_probs.append(argmax_prob)
        #             argmax_vocab_labels.append(argmax_label)
        #     argmax_probs = argmax_vocab_probs
        #     argmax_labels = argmax_vocab_labels
        # if self.block_vocab_labels is not None:
        #     argmax_vocab_probs = list()
        #     argmax_vocab_labels = list()
        #     for argmax_prob, argmax_label in zip(argmax_probs, argmax_labels):
        #         if argmax_label not in self.block_vocab_labels:
        #             argmax_vocab_probs.append(argmax_prob)
        #             argmax_vocab_labels.append(argmax_label)
        #     argmax_probs = argmax_vocab_probs
        #     argmax_labels = argmax_vocab_labels

        # watch_probs = []
        # watch_rankings = []
        # if watch_labels is not None:
        #     for watch_label in watch_labels:
        #         # watch_token = self.tokenizer.decode(watch_label)
        #         # logger.critical(f'{watch_label=}, {watch_token=}')
        #         # logger.critical(f'{torch.tensor(argmax_labels) == watch_label=}')
        #         ranking = torch.nonzero(torch.tensor(argmax_labels) == watch_label).flatten()
        #         prob = argmax_probs[ranking]
        #         watch_probs.append(prob)
        #         watch_rankings.append(int(ranking))

        # we only care about top tokens
        # argmax_probs = argmax_probs[:MAX_FOCUSING_NUM]
        # argmax_labels = argmax_labels[:MAX_FOCUSING_NUM]
        argmax_prob = argmax_probs[0]
        argmax_label = argmax_labels[0]
        return argmax_prob, argmax_label

    def inject(self, test_embeds, train_embeds):
        # train_embeds_nd = train_embeds.cpu().detach().numpy()
        # test_embeds_nd = test_embeds.cpu().detach().numpy()

        perplexity=30

        from openTSNE import initialization as initialization_scheme

        # If precomputed affinites are given, use those, otherwise proceed with
        # standard perpelxity-based affinites
        affinities = MultiscaleMixture(
            train_embeds,
            perplexity,
            metric="cosine",
            n_jobs=8,
            random_state=42,
            verbose=False,
        )

        # # PCA
        # pca_ = PCA(n_components=2, random_state=42)
        # embedding = pca_.fit_transform(train_embeds)
        # train_embeds = np.ascontiguousarray(embedding)

        # Y -= (np.max(Y, axis=0) + np.min(Y, axis=0)) / 2


        initialization = "median"
        k = 25

        affinity_params = {"perplexities": perplexity}

        P, neighbors, distances = affinities.to_new(
            test_embeds, return_distances=True, **affinity_params
        )

        # If initial positions are given in an array, use a copy of that
        if initialization == "weighted":
            embedding = initialization_scheme.weighted_mean(
                test_embeds, train_embeds, neighbors[:, :k], distances[:, :k]
            )
        elif initialization == "median":
            embedding = initialization_scheme.median(train_embeds, neighbors[:, :k])
        else:
            raise ValueError(f"Unrecognized initialization scheme `{initialization}`.")

        return embedding

    def _predict(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
        )
        logits = F.softmax(outputs.scores[0], dim=-1)
        return logits

    def optimize_trainer(self):
        train_embeds = self.train_embeds
        train_labels = self.train_labels
        test_embeds = self.test_embeds
        test_labels = self.test_labels

        # logger.debug(f'{len(train_embeds)=}')
        # logger.debug(f'{len(train_labels)=}')
        train_dataset = CoupleDataset(train_embeds, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.BATCH_SIZE)

        # logger.debug(f'{len(test_embeds)=}')
        # logger.debug(f'{len(test_labels)=}')
        test_dataset = CoupleDataset(test_embeds, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.BATCH_SIZE)

        # # ...
        # whole_dataset = CoupleDataset(train_embeds + test_embeds, train_labels + test_labels)
        # whole_loader = DataLoader(whole_dataset, batch_size=BATCH_SIZE)

        # dataset_size = len(dataset)
        # train_indices = list(range(0, int(dataset_size * 0.8)))
        # test_indices = list(range(int(dataset_size * 0.8), dataset_size))
        # train_sampler = SequentialSampler(train_indices)
        # test_sampler = SequentialSampler(test_indices)
        # train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=1)
        # test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=1)

        # xxx
        if CFG_ANCHOR_INIT == 'abs':
            self.learner.init_anchors()
        if CFG_ANCHOR_INIT == 'rel':
            self.learner.init_anchors(train_embeds, train_labels)

        train_embeds = torch.cat(train_embeds, 0)
        # logger.debug(f'{train_embeds.shape=}')
        if CFG_MODE_INSPECT:
            inspect(0, self.identifier, train_embeds, train_labels)

        test_embeds = torch.cat(test_embeds, 0)
        # logger.debug(f'{test_embeds.shape=}')
        # inspect(0, self.identifier, test_embeds, test_labels)

        if CFG_MODE_VERBOSE:
            # ...tSNE
            self.visualize(train_embeds, train_labels, test_embeds, test_labels, mark='old')

        # ...
        trainer = Trainer(
            max_epochs=self.cfg.EPOCH_NUM,
            precision='16' if DEVICE == 'cuda' else '32',
            # accelerator='cuda',
            # accelerator='mps',
            enable_progress_bar=True,
            enable_checkpointing=False,
            default_root_dir=f'../outs/{self.identifier}/',
        )
        trainer.fit(
            self.learner,
            train_loader,
            test_loader,
            # ckpt_path='../outs/tmp/'
        )

        # we freeze all parameters to save computation
        for name, parameter in self.learner.named_parameters():
            # print(name, parameter.size())
            parameter.requires_grad = False
        self.learner.eval()
        # trainer.validate(self.learner, dataloaders=couple_loader, verbose=False)
        # trainer.test(self.learner, dataloaders=couple_loader, verbose=False)

        if CFG_MODE_VERBOSE:
            logger.warning('TRAIN ...')
            train_refined_embeds = trainer.predict(self.learner, dataloaders=train_loader)
            train_refined_embeds = torch.cat(train_refined_embeds, 0).to(DEVICE)
            train_refined_embeds = train_refined_embeds.squeeze(1)
            # inspect(EPOCH_NUM, train_refined_embeds, train_labels)
            self.report_arc(train_embeds, train_refined_embeds, train_labels)

        logger.warning('TEST ...')
        test_refined_embeds = trainer.predict(self.learner, dataloaders=test_loader)
        test_refined_embeds = torch.cat(test_refined_embeds, 0).to(DEVICE)
        test_refined_embeds = test_refined_embeds.squeeze(1)
        # inspect(EPOCH_NUM, test_refined_embeds, test_labels)
        self.report_arc(test_embeds, test_refined_embeds, test_labels)

        # # ...
        # train_embeds_nd = train_embeds.cpu().detach().numpy()
        # test_embeds_nd = test_embeds.cpu().detach().numpy()
        # test_refined_embeds = self.prepare_partial(test_embeds_nd, train_embeds_nd)
        # # test_refined_embeds = torch.tensor(test_refined_embeds)

        # # ... after the affinity transformation
        # test_refined_embeds = self.inject(test_refined_embeds.cpu(), train_refined_embeds.cpu())
        # test_refined_embeds = torch.tensor(test_refined_embeds).to(DEVICE)
        # self.report_gen(test_embeds, test_refined_embeds, test_labels)

        if CFG_MODE_VERBOSE:
            # ...tSNE
            self.visualize(train_refined_embeds, train_labels, test_refined_embeds, test_labels, mark='new')

        # logger.debug(f'{train_embeds.shape=}')
        # logger.debug(f'{train_refined_embeds.shape=}')
        # logger.debug(f'{test_embeds.shape=}')
        # logger.debug(f'{test_refined_embeds.shape=}')

        # baseline GEN
        self.report_gen(test_embeds, test_refined_embeds, test_labels)
        if CFG_MODE_VERBOSE:
            # baseline RAG
            self.report_rag(train_embeds, train_refined_embeds, train_labels, test_embeds, test_refined_embeds, test_labels)

    def visualize(self, train_embeds, train_labels, test_embeds, test_labels, mark='tsne'):
        import arc_vis

        import matplotlib.pyplot as plt
        from openTSNE import TSNE

        folder = VISUAL_OUTS_DIR / self.identifier
        folder.mkdir(parents=True, exist_ok=True)

        tsne = TSNE(metric='cosine', n_jobs=8, random_state=42)
        embedding_train = tsne.fit(train_embeds.cpu())
        arc_vis.plot(embedding_train, train_labels, colors=arc_vis.MOUSE_10X_COLORS)
        plt.savefig(folder / f'tsne_{mark}_train.png', bbox_inches='tight', dpi=300)
        # plt.show()
        embedding_test = embedding_train.transform(test_embeds.cpu())
        arc_vis.plot(embedding_test, test_labels, colors=arc_vis.MOUSE_10X_COLORS)
        plt.savefig(folder / f'tsne_{mark}_test.png', bbox_inches='tight', dpi=300)
        # plt.show()

        # import umap
        #
        # manifold = umap.UMAP(metric='cosine', n_jobs=8, random_state=42).fit(train_embeds.cpu(), train_labels)
        # reduced_data = manifold.transform(train_embeds.cpu())
        # arc_vis.plot(reduced_data, train_labels, colors=arc_vis.MOUSE_10X_COLORS)
        # plt.savefig(f'umap_{mark}_train.png', bbox_inches='tight', dpi=300)
        # # plt.show()
        # reduced_data = manifold.transform(test_embeds.cpu())
        # arc_vis.plot(reduced_data, test_labels, colors=arc_vis.MOUSE_10X_COLORS)
        # plt.savefig(f'umap_{mark}_test.png', bbox_inches='tight', dpi=300)
        # # plt.show()

    # TODO
    #  GEN -> RAG -> KP-RAG -> VA-RAG
    #  VA-RAG -> VA-RAG(mini) [Johnson–Lindenstrauss 定理] 一个一百万维空间里的随便一万个点，一定可以几乎被装进一个几十维的子空间里
    #  https://www.zhihu.com/question/60648826/answer/2442192637
    # 1. model-predict (GEN)
    # 2. retrieval-aug (RAG)
    # 3. knn-prompting (KP-RAG)
    # 4. vocab-anchors (VA-GEN, VA-RAG)
    def infer(self, matrix, embeds):
        argmax_labels = list()
        # argmax_tokens = list()
        for embed in embeds:
            # embed -> logit -> label
            logits = torch.matmul(embed, matrix)
            logits = logits.unsqueeze(0)
            # logger.warning(f'{logits.shape=}')
            argmax_prob, argmax_label = self.logits_to_output(logits)
            argmax_labels.append(argmax_label)
        return argmax_labels

    def refer(self, anchors, embeds):
        argmax_labels = list()
        # argmax_tokens = list()
        for embed in embeds:
            # embed -> logit -> label
            simis = self.learner.simi_fn(embed, anchors)
            logits = nn.functional.softmax(simis, dim=-1).to(DEVICE)
            logits = logits.detach().unsqueeze(0)
            # logger.warning(f'{logits.shape=}')
            argmax_prob, argmax_label = self.logits_to_output(logits)
            argmax_labels.append(argmax_label)
        return argmax_labels

    def ragen(self, test_embeds, train_embeds, train_labels):
        # train_logits = self.embeds_to_logits(train_embeds)
        # anchors = train_logits
        anchors = train_embeds.squeeze(1)

        # TODO this is by COS
        def get_knn_idx(queries, anchors, k=1):
            # return idx of k nearest neighbours of the query
            # query: logits
            # neighbour_idx: [idx_1, ...]

            size = len(queries)
            # queries = torch.stack(queries, dim=0).to(DEVICE)
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

        ks = [2 ** exp for exp in range(11)]
        assert 1 in ks
        all_neighbors = get_knn_idx(test_embeds, anchors, k=ks[-1])

        all_answers = list()
        for neighbors in all_neighbors:
            # logger.warning(f'{neighbors=}')
            answers = list()
            for k in ks:
                top_neighbors = neighbors[:k]
                top_answers = [train_labels[neighbor] for neighbor in top_neighbors]
                counter = Counter(top_answers)
                answer = counter.most_common(1)
                answers.append(answer[0][0])
            all_answers.append(answers)
        return all_answers

    def report_arc(self, origin_embeds, reform_embeds, refer_labels):
        # ...
        anchors = self.learner.anchors
        gen_labels = self.refer(anchors.to(DEVICE), origin_embeds.to(DEVICE))
        arc_labels = self.refer(anchors.to(DEVICE), reform_embeds.to(DEVICE))
        Metric.contrast_gen_scoring(gen_labels, arc_labels, refer_labels)
        # dependent on the REFER practice
        Metric.contrast_cluster_measuring(gen_labels, arc_labels, refer_labels)

        # ...
        ada_anchors = self.learner.adapt_anchors(anchors)
        ada_labels = self.refer(ada_anchors.to(DEVICE), origin_embeds.to(DEVICE))
        Metric.contrast_gen_scoring(gen_labels, ada_labels, refer_labels)
        # dependent on the REFER practice
        Metric.contrast_cluster_measuring(gen_labels, ada_labels, refer_labels)

        # dump generalizations
        # prefix = '_' if DATA_NAME == 'human' else ''
        # well_dump_gens(prefix + '_'.join([DATA_NAME, EXP_CODE]), all_pre_gens, all_post_gens, all_oracle_gens)
        # well_dump_gens('_'.join([DATA_NAME, EXP_CODE]), all_pre_gens, all_post_gens, all_oracle_gens)

    def report_gen(self, origin_embeds, reform_embeds, refer_labels):
        # ...
        matrix = self.learner.build_actual_matrix()
        gen_labels = self.infer(matrix, origin_embeds)
        arc_labels = self.infer(matrix, reform_embeds)
        Metric.contrast_gen_scoring(gen_labels, arc_labels, refer_labels)

    def report_rag(
            self,
            train_origin_embeds,
            train_reform_embeds,
            train_labels,
            test_origin_embeds,
            test_reform_embeds,
            test_labels
    ):
        all_origin_answers = self.ragen(test_origin_embeds, train_origin_embeds, train_labels)
        all_reform_answers = self.ragen(test_reform_embeds, train_reform_embeds, train_labels)

        ks = [2 ** exp for exp in range(11)]
        for idx, k in enumerate(ks):
            origin_answers = [answer[idx] for answer in all_origin_answers]
            reform_answers = [answer[idx] for answer in all_reform_answers]
            # logger.error(f'{(0 in origin_answers)=}')
            # logger.error(f'{(0 in reform_answers)=}')
            # logger.error(f'{(0 in test_labels)=}')
            # logger.error(f'{(20 in origin_answers)=}')
            # logger.error(f'{(20 in reform_answers)=}')
            # logger.error(f'{(20 in test_labels)=}')
            # logger.info(f'{k=} {origin_answers[:10]=}')
            # logger.info(f'{k=} {reform_answers[:10]=}')
            # logger.info(f'{k=} {test_labels[:10]=}')
            # ...
            Metric.contrast_gen_scoring(origin_answers, reform_answers, test_labels)
