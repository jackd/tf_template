from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class InferenceModel(object):
    def get_inference(self, features, mode):
        """
        Get the inference associated with this modelself.

        This should create all variables necessary. For example, for
        classification problems, this would be the logits.

        Args:
            features: possibly nested structure of tensors returned by
                the data source's `get_inputs` (first return value).
            mode: on of `tf.estimator.ModeKeys` - 'train', 'eval' or 'infer'.

        Returns:
            possibly nested structure of tensors for predictions/losses.
        """
        raise NotImplementedError('Abstract method')

    def get_predictions(self, features, inference):
        """
        Convert inferences to predictions.

        This should not introduce new trainable parameters.

        Args:
            features: first output of `DataSource.get_inputs` - possibly nested
                structure of batched tensors.
            inference: output of `self.get_inference`.

        Returns:
            possibly nested structure of tensor predictions.

        Defaults to returning inference unchanged.
        """
        return inference

    def prediction_vis(self, prediction_data):
        """
        Get a vis of prediction data for a single example.

        Args:
            prediciton_data: numpy data with same structure as
                `self.get_predictions` output.

        Returns:
            `Visualization`, or iterable of `Visualization`s
        """
        raise NotImplementedError('Abstract method')

    def get_warm_start_settings(self):
        """
        Get `tf.estimator.WarmStartSettings` for transfer learning.

        Can be None, in which case variables are initialized from scratch.
        """
        return None


class DelegatingInferenceModel(InferenceModel):
    """
    Wrapper class that defaults to redirecting all methods to another model.

    Derived classes presumably override some methods.
    """
    def __init__(self, base):
        self._base = base

    @property
    def base(self):
        return self._base

    def get_inference(self, features, mode):
        return self._base.get_inference(features, mode)

    def get_predictions(self, features, inference):
        return self._base.get_predictions(features, inference)

    def prediction_vis(self, prediction_data):
        return self._base.prediction_vis(prediction_data)

    def get_warm_start_settings(self):
        return self._base.get_warm_start_settings()


class TransferInferenceModel(DelegatingInferenceModel):
    def __init__(self, base, warm_start_settings):
        self._warm_start_settings = warm_start_settings
        super(TransferInferenceModel, self).__init__(base)

    def get_warm_start_settings(self):
        return self._warm_start_settings
