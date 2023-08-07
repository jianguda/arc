from arc_shared import MODEL_REGISTRY


class Configure:
    def __init__(self, args):
        self.DATA_CODE = args.data
        self.MODEL_CODE = args.model
        self.MODEL_NAME = MODEL_REGISTRY[args.model]

        self.EPOCH_NUM = args.epoch_num
        self.BATCH_SIZE = args.batch_size

        self._init()
        self._refresh(args.exp)

    def _init(self):
        # approach, baseline
        self.OPT_STANDARD = 'approach'
        # noat, noca, noma, normal
        self.OPT_ABLATION = 'normal'
        # (decay) linear, exp, log
        self.OPT_HEURISTIC = 'linear'

    def _refresh(self, exp_code: str):
        exp_details = exp_code.split('.')
        if len(exp_details) != 3:
            return None
        exp_type, exp_key, exp_value = exp_details

        assert exp_type in ['standard', 'ablation', 'heuristic', 'saliency']
        if exp_type == 'standard':
            assert exp_value in ['approach', 'baseline']
            self.OPT_STANDARD = exp_value
        if exp_type == 'ablation':
            assert exp_value in ['noat', 'noca', 'noma', 'normal']
            self.OPT_ABLATION = exp_value
        if exp_type == 'heuristic':
            assert exp_value in ['mm4', 'tab', 'type', 'linear', 'exp', 'log']
            self.OPT_HEURISTIC = exp_value
