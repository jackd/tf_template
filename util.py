from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six


def maybe_stop():
    """Prompt user to continue or stop."""
    inp = six.moves.input('Continue? (Y/n): ').lower().rstrip()
    if inp == 'n':
        # exit()
        raise Exception('Stopped')
