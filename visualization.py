"""
Provides `Visualization` interface and common implementations.

A `Visualization` is something that can be shown (with optional blocking) and
closed.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Visualization(object):
    """
    Something that can be shown and closed on demand.

    Used for combining visualizations of different kinds, e.g. mayavi
    figures and plt figures.

    See implementations for example usage.
    """

    def show(self, block=False):
        raise NotImplementedError('Abstract method')

    def close(self):
        raise NotImplementedError('Abstract method')

    def __call__(self, block=False):
        return self.show(block=block)


class CompoundVis(Visualization):
    def __init__(self, *visualizations):
        self._visualizations = visualizations

    @property
    def visualizations(self):
        return self._visualizations

    def show(self, block=False):
        for v in self._visualizations[:-1]:
            v.show(block=False)
        self._visualizations[-1].show(block=block)
        if block:
            for v in self._visualizations[:-1]:
                v.close()

    def close(self):
        for v in self._visualizations:
            v.close()


class PltVis(Visualization):
    """Generic Visualization based on matplotlib.pyplot."""
    def show(self, block=False):
        import matplotlib.pyplot as plt
        plt.show(block=block)

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()


class ImageVis(PltVis):
    def __init__(self, image, **imshow_kwargs):
        import numpy as np
        import matplotlib.pyplot as plt
        self._fig, self._ax = plt.subplots(1, 1)
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        self._ax.imshow(image, **imshow_kwargs)


class MultiImageVis(PltVis):
    def __init__(self, images, *grid_shape, **imshow_kwargs):
        import numpy as np
        import matplotlib.pyplot as plt
        assert(np.prod(grid_shape) == len(images))
        self._fig, self._ax = plt.subplots(*grid_shape)
        for ax, image in zip(self._ax.flatten(), images):
            ax.imshow(image, **imshow_kwargs)


class PrintVis(Visualization):
    """Default Visualization that prints data to screen."""

    def __init__(self, data):
        self._data = data

    def show(self, block=False):
        from .util import get_input
        print(self._data)
        if block:
            get_input('Enter to continue...')

    def close(self):
        pass


def get_vis(*vis):
    """
    Wrap different data in default `Visualization`sself.

    2 or 3D numpy array: imshow
    object with a `vis` and `close` attribute: itself
    non-string iterable: CompoundVis
    otherwise: PrintVis
    """
    import numpy as np
    if len(vis) == 1:
        vis = vis[0]
    if isinstance(vis, np.ndarray):
        n_dims = len(vis.shape)
        if n_dims in (2, 3):
            # assumed image
            return ImageVis(vis)
        else:
            raise ValueError(
                'No default vis for ndarray of shape %s' % vis.shape)
    elif hasattr(vis, 'show') and hasattr(vis, 'close'):
        return vis
    elif isinstance(vis, (str, unicode)):
        return PrintVis(vis)
    elif hasattr(vis, '__iter__'):
        vis = tuple(get_vis(v) for v in vis)
        return CompoundVis(*vis)
    else:
        return PrintVis(vis)
