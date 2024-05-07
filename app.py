import gradio as gr
import numpy as np
import cv2
import tensorflow as tf

from scripts.utils import norm_crop, estimate_norm, inverse_estimate_norm, transform_landmark_points, get_lm
from scripts.layers import AdaIN, AdaptiveAttention
from scripts.models import FPN, SSH, BboxHead, LandmarkHead, ClassHead

from tensorflow_addons.layers import InstanceNormalization
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model

class FaceMaker:
    def __init__(self):
        self.RetinaFace = load_model("./models/retinaface.ism", custom_objects={"FPN": FPN, "SSH": SSH, "BboxHead": BboxHead, "LandmarkHead": LandmarkHead, "ClassHead": ClassHead})
        self.ArcFace = load_model("./models/arcface.ism")
        self.G = load_model("./models/facemaker.ism", custom_objects={"AdaIN": AdaIN, "AdaptiveAttention": AdaptiveAttention, "InstanceNormalization": InstanceNormalization})

    def swap(self, target, source):
        try:
            source = np.array(source)
            target = np.array(target)

            # prepare blend mask
            blend_mask_base = np.zeros(shape=(256, 256, 1))
            blend_mask_base[80:244, 32:224] = 1
            blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

            # prepare to load image
            source_a = self.RetinaFace(np.expand_dims(source, axis=0)).numpy()[0]
            source_h, source_w, _ = source.shape
            source_lm = get_lm(source_a, source_w, source_h)
            source_aligned = norm_crop(source, source_lm, image_size=256)
            source_z = self.ArcFace.predict(np.expand_dims(tf.image.resize(source_aligned, [112, 112]) / 255.0, axis=0))

            # read frame
            im = target
            im_h, im_w, _ = im.shape
            im_shape = (im_w, im_h)

            detection_scale = im_w // 640 if im_w > 640 else 1

            faces = self.RetinaFace(np.expand_dims(cv2.resize(im, (im_w // detection_scale, im_h // detection_scale)), axis=0)).numpy()

            total_img = im / 255.0

            for annotation in faces:
                lm_align = np.array([[annotation[4] * im_w, annotation[5] * im_h],
                                     [annotation[6] * im_w, annotation[7] * im_h],
                                     [annotation[8] * im_w, annotation[9] * im_h],
                                     [annotation[10] * im_w, annotation[11] * im_h],
                                     [annotation[12] * im_w, annotation[13] * im_h]],
                                    dtype=np.float32)

                # align the detected face
                M, pose_index = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
                im_aligned = (cv2.warpAffine(im, M, (256, 256), borderValue=0.0) - 127.5) / 127.5

                # face swap
                changed_face_cage = self.G.predict([np.expand_dims(im_aligned, axis=0), source_z])
                changed_face = changed_face_cage[0] * 0.5 + 0.5

                # get inverse transformation landmarks
                transformed_lmk = transform_landmark_points(M, lm_align)

                # warp image back
                iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
                iim_aligned = cv2.warpAffine(changed_face, iM, im_shape, borderValue=0.0)

                # blend swapped face with target image
                blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
                blend_mask = np.expand_dims(blend_mask, axis=-1)
                total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))

            total_img = np.clip(total_img, 0, 1)
            total_img *= 255.0
            total_img = total_img.astype('uint8')

            return total_img

        except Exception as e:
            print(e)
            return None
    
    def ui(self):
        with gr.Blocks(analytics_enabled=False, title='FaceMaker', theme=gr.themes.Monochrome(), css='footer {display: none !important;}') as fm:
            header = gr.HTML(value='<center><h3>STELLA FaceMaker by Ikmal Said (@ikmalsaid)</h3></center>')
            with gr.Row():
                target = gr.Image(type="pil", label='Target')
                source = gr.Image(type="pil", label='Source')
                output = gr.Image(type="pil", label='Output')
            swap = gr.Button("Face Swap").click(self.swap, inputs=[target, source], outputs=output)
        fm.launch(inbrowser=True)

if __name__ == "__main__":
    facemaker = FaceMaker()
    facemaker.ui()