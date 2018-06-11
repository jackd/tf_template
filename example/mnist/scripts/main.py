#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
from tf_template.cli import coord_main

flags.DEFINE_string('model_id', default='base', help='id of model')


def main(_):
    from tf_template.example.mnist.coordinator import get_mnist_coordinator
    coord_main(get_mnist_coordinator(flags.FLAGS.model_id))


app.run(main)
