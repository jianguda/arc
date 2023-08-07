from pathlib import Path

from transformers import (
    AutoConfig, GPT2LMHeadModel, GPTNeoXForCausalLM,
)


def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name, output_scores=True)
    # config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_scores=True)

    if "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        print(model)
    elif "pythia" in model_name:
        model = GPTNeoXForCausalLM.from_pretrained(model_name, config=config)
    else:
        raise NotImplementedError
    return model


def dump_model_info(model_tag, model_repr):
    folder = Path('../docs')
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f'{model_tag}.txt', 'w') as handle:
        handle.writelines(model_repr)


def main():
    """
    Models                          Parameters  Layers  Dimension
    gpt2                            117M        12      768
    gpt2-medium                     345M        24      1024
    gpt2-large                      762M        36      1280
    gpt2-xl                         1542M       48      1600
    EleutherAI/pythia-70m-deduped   70M         6       512
    EleutherAI/pythia-160m-deduped  160M        12      768
    EleutherAI/pythia-410m-deduped  410M        24      1024
    EleutherAI/pythia-1b-deduped    1B          16      2048
    EleutherAI/pythia-1.4b-deduped  1.4B        24      2048
    EleutherAI/pythia-2.8b-deduped  2.8B        32      2560
    """
    gpt2_model_names = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    pythia_model_names = [
        'EleutherAI/pythia-70m-deduped',
        'EleutherAI/pythia-160m-deduped',
        'EleutherAI/pythia-410m-deduped',
        'EleutherAI/pythia-1b-deduped',
        'EleutherAI/pythia-1.4b-deduped',
        'EleutherAI/pythia-2.8b-deduped',
    ]
    model_names = gpt2_model_names + pythia_model_names
    for model_name in model_names:
        model = load_model(model_name)
        model_tag = model_name.split('/')[-1]
        model_repr = repr(model)
        dump_model_info(model_tag, model_repr)
        break


if __name__ == '__main__':
    main()
