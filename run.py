from utils import *
from progressbar import ProgressBar
from config import OBJECTS_PER_IMAGE_RANGE
import random

if not os.path.isdir(cfg.RESULT_DIR):
    os.makedirs(cfg.RESULT_DIR)
if not os.path.isdir(cfg.BACKGROUND_DIR):
    raise Exception("Wrong BACKGROUND_DIR")
if not os.path.isdir(cfg.OBJECTS_DIR):
    raise Exception("Wrong OBJECTS_DIR")

ep_dir = get_new_epoch_path()
os.makedirs(ep_dir)
if cfg.MERGE_OUTPUTS:
    data_path = ep_dir
    out_path = ep_dir
else:
    data_path = ep_dir + "data/"
    out_path = ep_dir + "label/"
    os.makedirs(data_path)
    os.makedirs(out_path)

bg_names = os.listdir(cfg.BACKGROUND_DIR)

obj_images = get_objects()

with ProgressBar(max_value=get_amount(obj_images, bg_names)) as bar:
    name_amount = 0
    for bg_name in bg_names:
        bg = cv2.imread(cfg.BACKGROUND_DIR + bg_name)

        curr_data_path = data_path
        curr_out_path = out_path
        if cfg.PACKAGE_BY_BACKGROUND:
            curr_data_path += bg_name.replace(".", "__") + "/"
            curr_out_path += bg_name.replace(".", "__") + "/"
            os.makedirs(curr_data_path)
            if not cfg.MERGE_OUTPUTS:
                os.makedirs(curr_out_path)

        for i in range(len(obj_images)):
            for _ in range(cfg.IMAGES_PER_COMBINATION):
                edited_img = bg
                for j, el in enumerate(random.sample(obj_images, random.randint(*OBJECTS_PER_IMAGE_RANGE))):
                    img, out = generate_img(edited_img, el)
                    edited_img = img
                    if cfg.OUTPUT_FORMAT == "classification":
                        save_as_txt(out, curr_out_path + str(name_amount))
                    elif cfg.OUTPUT_FORMAT == "segmentation":
                        if not os.path.isdir(curr_out_path + str(name_amount)):
                            os.makedirs(curr_out_path + str(name_amount))
                        save_as_grayscale_img(out, curr_out_path + str(name_amount) + '/' + f'{j}.jpg')
                blured_img = add_blur(img.astype(np.float32), cfg.BLUR_PROBABILITY, cfg.BLUR_KERNEL_RANGE,
                                      cfg.BLUR_INTENSITY_RANGE).astype(np.uint8)
                noised_img = add_noise(img, cfg.NOISE_LEVEL_RANGE)
                cv2.imwrite(curr_data_path + str(name_amount) + ".jpg", noised_img)
                name_amount += 1
                bar.update(name_amount)
