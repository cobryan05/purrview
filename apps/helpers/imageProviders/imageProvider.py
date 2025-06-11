''' Image Provider Interface Class '''

import numpy as np


class ImageProvider:
    async def getNextImage(self) -> np.array:
        """Retrieves the next image for processing from the image source"""
        raise NotImplementedError()
