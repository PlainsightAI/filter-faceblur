import cv2
import numpy as np

from filter_faceblur.model.blurrers.base_blurrer import BaseBlurrer


class MedianBlur(BaseBlurrer):
    def blur(self, image, face, blur_strength=1.0):
        (x, y, w, h) = face
        face_region = image[y : y + h, x : x + w]
        if face_region.size == 0:
            return image

        base_kernel_size = 99

        # cv2.medianBlur requires an odd kernel size >= 3.
        kernel_size = max(3, int(base_kernel_size * blur_strength))
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred_face = cv2.medianBlur(face_region, kernel_size)
        mask = np.zeros_like(face_region[..., :1], dtype=np.uint8)
        cntr = (w // 2, h // 2)
        cv2.ellipse(mask, cntr, (w // 2, h // 2), 0, 0, 360, 1, -1)
        image[y : y + h, x : x + w] = np.where(mask, blurred_face, face_region)
        return image

    def get_name(self):
        return self.__class__.__name__

    def get_params(self):
        return {}
