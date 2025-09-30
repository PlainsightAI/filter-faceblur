import cv2, os
from filter_faceblur.model import FaceBlur

MODEL_ARTIFACTORY_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
MODEL_ARTIFACT_NAME="face_detection_yunet_2023mar.onnx"

def main():

    model_artifact = os.path.join(MODEL_ARTIFACTORY_URL, MODEL_ARTIFACT_NAME)
    face_blur = FaceBlur(model_artifact, detector_name="yunet", blurrer_name="gaussian")

    input_image_path = "data/people.jpg"
    output_image_path = "data/people_blurred.jpg"

    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Failed to read frame {input_image_path}")

    processed_image = face_blur.process_frame(image)
    cv2.imwrite(output_image_path, processed_image)

if __name__ == "__main__":
    main()

