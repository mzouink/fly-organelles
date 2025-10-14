import gunpowder as gp
import numpy as np
from fly_organelles.lsds.lite.lsds import get_lsds as get_lsds_lite
from fly_organelles.lsds.lite.affs import get_affs as get_affs_lite
import logging

logger = logging.getLogger(__name__)


class LSDAffinities(gp.BatchFilter):

    def __init__(self, array, affinities, sigma=2.0, voxel_size=None):
        self.array = array
        self.affinities = affinities
        self.sigma = sigma
        self.voxel_size = voxel_size
        self.affinities = affinities

    def setup(self):
        self.spec[self.array].dtype = np.float32

    def process(self, batch, request):

        data = batch[self.array].data
        data = data.astype(np.uint8)
        if not np.any(data):
            lsd_data = np.zeros((10 + len(self.affinities),) + data.shape, dtype=np.float32)
            # logger.error(f"LSD empty shape: {lsd_data.shape}, dtype: {lsd_data.dtype}")
        else:
            lsd_data = get_lsds_lite(data, sigma=self.sigma, voxel_size=self.voxel_size).astype(np.float32)
            affs_data = get_affs_lite(data, neighborhood=self.affinities, dist="equality-no-bg", pad=True).astype(
                np.float32
            )
            lsd_data = np.concatenate([affs_data, lsd_data], axis=0)
            # logger.error(f"LSD shape: {lsd_data.shape}, dtype: {lsd_data.dtype}")
        batch[self.array].data = lsd_data
        # batch[self.array].dtype = np.float32
