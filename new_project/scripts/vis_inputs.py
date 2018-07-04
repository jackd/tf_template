#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
from tf_template.cli import vis_inputs
FLAGS = flags.FLAGS

# TODO
flags.DEFINE_string(
    'cat_id', default=None,
    help='categroy id')


def main(_):
    from new_project.data_source import get_data_source
    # TODO
    data_source = get_data_source(some_float=FLAGS.cat_id)
    vis_inputs(data_source)


app.run(main)
