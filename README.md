# SyntheticData Generator

With this program, you can generate synthetic data to train your AI in image detection.

---

## How to use

Open config, set your result, bg, obj PATHs, set parameters as you like and run `run.py` file. It will generate all possible combinations of bg and object.

### Parameters

## Configuration Parameters

### Paths
- `RESULT_DIR`: Directory where generated images will be saved.  
  Example: `"Result/"`

- `BACKGROUND_DIR`: Directory containing background images.  
  Example: `"Bg/"`

- `OBJECTS_DIR`: Directory containing object images.  
  Example: `"Obj/"`

### Core Parameters
- `IGNORE_FILENAME_SYMBOL`: Characters in filenames that should be ignored when loading objects.  
  Example: `"!!!!"`

- `MERGE_OUTPUTS`: Whether to merge data and labels into a single folder.  
  Values: `True` or `False`

- `PACKAGE_BY_BACKGROUND`: Whether to create separate folders for each background.  
  Values: `True` or `False`

- `ALLOW_OUT_OF_BOUNDS`: Allow objects to be placed partially outside the image bounds.  
  Values: `True` or `False`

- `OUT_OF_BOUNDS_RANGE`: Range for how much objects can be placed out of bounds.  
  Example: `(0.1, 0.4)`

- `IMAGES_PER_COMBINATION`: Number of images generated per background-object combination.  
  Example: `5`

### Object Parameters
- `OBJECTS_PER_IMAGE_RANGE`: Range for the number of objects per image.  
  Example: `(1, 3)`

- `PLACEMENT_DISTRIBUTION`: Object placement distribution, either `"uniform"` or `"gaussian"`.  
  Example: `"gaussian"`

- `OBJECT_SCALE_RANGE`: Range for scaling objects.  
  Example: `(0.5, 1.5)`

- `BLUR_PROBABILITY`: Probability of blurring objects.  
  Example: `0.7`

- `BLUR_TYPE`: Type of blur, either `"GaussianBlur"` or `"bilateralFilter"`.  
  Example: `"bilateralFilter"`

- `BLUR_KERNEL_RANGE`: Range for the blur kernel size.  
  Example: `(0.7, 1.0)`

- `BLUR_INTENSITY_RANGE`: Range for blur intensity.  
  Example: `(20, 70)`

- `NOISE_TYPE`: Type of noise added, either `"uniform"` or `"gaussian"`.  
  Example: `"gaussian"`

- `NOISE_LEVEL_RANGE`: Range for noise intensity.  
  Example: `(0.1, 0.2)`

- `FLIP_PROBABILITY`: Probability of flipping objects horizontally.  
  Example: `0.5`

The `scale_rate`, `blur_rate` and `noise_rate` parameters must be given as a tuple with two values `(a, b)` for a random value from the range, list/int/float `[a, b, c]` / ` d` for a random specific value.

---

### Cropping your objects

You can run `crop.py` to crop the empty edges of your objects from the obj folder you specified in `config.py`. This will create a new folder with all your cropped objects. Delete/rename the old "Cropped" folder before running the script again.
