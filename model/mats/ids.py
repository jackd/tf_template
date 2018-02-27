import random


def get_example_ids(cat_id, mode):
    from shapenet.core import get_example_ids
    example_ids = list(get_example_ids(cat_id))
    random.seed(0)
    random.shuffle(example_ids)
    n = int(0.8*len(example_ids))
    if mode == 'train':
        example_ids = example_ids[:n]
    else:
        example_ids = example_ids[n:]
    return example_ids
