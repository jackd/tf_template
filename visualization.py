from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Visualization(object):

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


class ImageVis(Visualization):
    def __init__(self, image):
        import numpy as np
        import matplotlib.pyplot as plt
        self._fig, self._ax = plt.subplots(1, 1)
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        self._ax.imshow(image)

    def show(self, block=False):
        import matplotlib.pyplot as plt
        plt.show(block=block)

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()


class PrintVis(Visualization):
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
