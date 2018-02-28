from voxel_ae import register_voxel_ae_family
from image_encoder import register_image_encoder_family
from fine_tune import register_fine_tune_family


def register_all():
    register_voxel_ae_family()
    register_image_encoder_family()
    register_fine_tune_family()
