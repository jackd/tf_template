from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    get_input = raw_input
except NameError:
    get_input = input


def maybe_stop():
    """Prompt user to continue or stop."""
    inp = get_input('Continue? (Y/n): ').lower().rstrip()
    if inp == 'n':
        exit()
