#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
from tf_template.cli import vis_inputs
FLAGS = flags.FLAGS

flags.DEFINE_float(
    'corruption', default=None,
    help='stddev or corruption applied to inputs')


def main(_):
    from tf_template.example.mnist.data_source import MnistDataSource
    data_source = MnistDataSource(corruption_stddev=flags.FLAGS.corruption)
    vis_inputs(data_source)


app.run(main)
