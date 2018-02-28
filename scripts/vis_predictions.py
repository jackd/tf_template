
def main(model_id, mode):
    import tensorflow as tf
    from tf_template.model import get_builder
    tf.logging.set_verbosity(tf.logging.INFO)
    builder = get_builder(model_id)
    builder.vis_predictions(mode)


if __name__ == '__main__':
    import argparse
    from tf_template.setup import register_families

    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help='id of model defined in params')
    parser.add_argument(
        '-m', '--mode', help='mode for sourcing data', default='infer')
    args = parser.parse_args()

    register_families()
    main(args.model_id, args.mode)
