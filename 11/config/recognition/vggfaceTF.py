from __future__ import print_function
from recognition.modelsTF import RESNET50

def VGGFace(include_top=True, model='resnet50', weights='vggface',
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=None,weights_path=None):

    if weights not in {'vggface', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `vggface`'
                         '(pre-training on VGGFace Datasets).')


    if model == 'resnet50':

        if classes is None:
            classes = 8631

        if weights == 'vggface' and include_top and classes != 8631:
            raise ValueError(
                'If using `weights` as vggface original with `include_top`'
                ' as true, `classes` should be 8631')

        return RESNET50(include_top=include_top, input_tensor=input_tensor,
                        input_shape=input_shape, pooling=pooling,
                        weights=weights,
                        classes=classes,weights_path=weights_path)