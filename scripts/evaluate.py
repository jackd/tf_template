
def main(model_id):
    import tensorflow as tf
    from tf_template.model import get_builder
    tf.logging.set_verbosity(tf.logging.INFO)
    builder = get_builder('example')
    evaluation = builder.eval()
    print(evaluation)


if __name__ == '__main__':
    import argparse
    from tf_template.setup import register_families
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help='id of model defined in params')
    args = parser.parse_args()

    register_families()
    main(args.model_id)
