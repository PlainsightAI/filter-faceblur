import cv2
import numpy as np

from filter_faceblur.model.blurrers.base_blurrer import BaseBlurrer


class GaussianBlur(BaseBlurrer):
    def blur(self, image, face, blur_strength=1.0):
        (x, y, w, h) = face
        face_region = image[y : y + h, x : x + w]
        if face_region.size == 0:
            return image
        
        # Calculate blur parameters based on strength
        # Base kernel size and sigma, scaled by blur_strength
        base_kernel_size = 99
        base_sigma = 30
        
        # Scale kernel size and sigma by blur_strength
        kernel_size = max(1, int(base_kernel_size * blur_strength))
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = base_sigma * blur_strength
        
        blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), sigma)
        mask = np.zeros_like(face_region[..., :1], dtype=np.uint8)
        cntr = (w // 2, h // 2)
        cv2.ellipse(mask, cntr, (w // 2, h // 2), 0, 0, 360, 1, -1)
        image[y : y + h, x : x + w] = np.where(mask, blurred_face, face_region)
        return image

    def get_name(self):
        return self.__class__.__name__

    def get_params(self):
        return {}
