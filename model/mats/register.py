from voxel_ae import register_voxel_ae_family
from image_encoder import register_image_encoder_family


def register_all():
    register_voxel_ae_family()
    register_image_encoder_family()
