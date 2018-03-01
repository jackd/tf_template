import os
from dataset.hdf5 import Hdf5Dataset

encodings_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), '_encodings')


def _get_embeddings_dataset(encoder_id, mode='r'):
    if not os.path.isdir(encodings_dir):
        os.makedirs(encodings_dir)
    path = os.path.join(encodings_dir, '%s.hdf5' % encoder_id)
    return Hdf5Dataset(path, mode)


def get_embeddings_dataset(encoder_id, mode='r'):
    dataset = _get_embeddings_dataset(encoder_id, mode)
    if not os.path.isfile(dataset.path):
        print(
            'No embeddings data available for %s - generating...' % encoder_id)
        create_embeddings_dataset(encoder_id)
    return dataset


def get_embeddings(encoder_id, mode='train'):
    from voxel_ae import VoxelAEBuilder
    builder = VoxelAEBuilder.from_id(encoder_id)
    for prediction in builder.predict(mode=mode):
        example_id = prediction['example_id']
        encoding = prediction['encoding']
        yield example_id, encoding


def create_embeddings_dataset(encoder_id, mode='train', overwrite=False):
    from tf_template.model import load_params
    from shapenet.core import cat_desc_to_id
    from ids import get_example_ids
    from shapenet.util import LengthedGenerator
    cat_desc = load_params(encoder_id)['cat']
    cat_id = cat_desc_to_id(cat_desc)
    example_ids = get_example_ids(cat_id, mode)
    embeddings = get_embeddings(encoder_id, mode)
    embeddings = LengthedGenerator(embeddings, len(example_ids))
    with _get_embeddings_dataset(encoder_id, 'a') as dataset:
        dataset.save_items(embeddings, overwrite=overwrite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help='id of model defined in params')
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-m', '--mode', type=str, default='train')
    args = parser.parse_args()

    create_embeddings_dataset(
        args.model_id, mode=args.mode, overwrite=args.overwrite)
