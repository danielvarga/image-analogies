import os
import time
import copy
import math

import numpy as np
from scipy.misc import imsave, imresize

from image_analogy import img_utils
import image_analogy.main


def create_mask(image, offset):
    height, width, ch = image.shape
    assert ch==3, 'RGBA input unexpected here'
    alpha = np.zeros_like(image[:, :, :1])
    alpha[:, :offset, :] = 255
    im_alpha = np.concatenate([image, alpha], axis=-1)
    return im_alpha


def blend(im1, im2):
    assert im1.shape==im2.shape
    height, width = im1.shape[:2]
    middle = width // 2
    col_weights = []
    for i in range(width):
        x = float(i-middle) / width * math.pi
        w = (math.sin(-x) + 1.0) / 2
        col_weights.append(w)
    col_weights = np.array(col_weights)
    weights = col_weights[np.newaxis, :, np.newaxis]

    bim1 = np.zeros((height, width, 3))
    bim1[:, :im1.shape[1], :] = im1
    bim2 = np.zeros((height, width, 3))
    bim2[:, -im2.shape[1]:, :] = im2
    bim = im1 * weights + im2 * (1.0 - weights)
    return bim


def main(args, model_class):
    assert args.consistency_image_path is None, 'Slicing can not be combined with pixelspace mask.'
    assert args.consistency_weight != 0.0, '--consistency-w argument missing, required for slicing images. Recommended value 1e8.'
    assert not((args.out_width > 0) or (args.out_height > 0)), 'Please don\'t combine --slices with the --width and --height arguments.'

    full_b_image = img_utils.load_image(args.b_image_path)
    height, full_width = full_b_image.shape[:2]
    N = args.slices

    # Trims away a border of this size from each output sub-image,
    # because borders of output images tend to be ugly.
    TRIMMING = 10 # Maybe some multiple of args.patch_size would be nicer.

    # Ad hoc, sure, but it should not matter too much.
    OVERLAP_RADIUS = min((50, max((TRIMMING, full_width // N // 20))))

    lefts = [0]
    rights = []
    for i in range(1, N):
        middle = full_width * i // N
        left   = middle - OVERLAP_RADIUS - TRIMMING
        right  = middle + OVERLAP_RADIUS + TRIMMING
        lefts.append(left)
        rights.append(right)
    rights.append(full_width)
    intervals = zip(lefts, rights)

    full_output_image = np.zeros_like(full_b_image)

    for indx, (left, right) in enumerate(intervals):
        print('Processing image slice B[{}:{}]'.format(left, right))
        assert left < right, 'Pathological combination of B image size and --slices argument value.'
        sub_b_image = full_b_image[:, left:right, :]
        sub_b_image_path = args.result_prefix + '_b_{}.png'.format(indx)
        print('Saving slice B image {}'.format(sub_b_image_path))
        imsave(sub_b_image_path, sub_b_image)

        sub_args = copy.copy(args)
        sub_args.slices = None
        sub_args.b_image_path = sub_b_image_path
        sub_args.result_prefix = args.result_prefix + '_{}'.format(indx)

        if indx > 0:
            l2, r2 = left, right
            l1, r1 = intervals[indx-1]

            # The idea is that if previous output subimage is B'1, and current subimage is B2,
            # then B'1[:, l2-l1:], B2[:, :r1-l2], and B_complete[:, l2:r1] should correspond.

            # This is complicated a bit by the fact that B1 and B1' (and thus B'1 and B2)
            # can have significantly differing dimensions, both in width and in height.
            # We paper over this by resizing B' to the shape of B.

            # A further minor complication is trimming: we don't want to use the last
            # TRIMMING columns of the outputs.

            consistency_mask_image = np.zeros_like(sub_b_image)
            consistency_mask_image[:, :r1-l2, :] = previous_sub_image[:, l2-l1:, :]
            offset = r1 - l2 - TRIMMING
            consistency_mask_alpha_image = create_mask(consistency_mask_image, offset)

            assert sub_b_image.shape[:2] == consistency_mask_alpha_image.shape[:2]

            sub_args.consistency_image_path = args.result_prefix + '_c_{}.png'.format(indx-1)
            print('Saving consistency mask {}'.format(sub_args.consistency_image_path))
            imsave(sub_args.consistency_image_path, consistency_mask_alpha_image)
        else:
            sub_args.consistency_weight = 0.0

        image_analogy.main.main(sub_args, model_class)

        # A proper output filename would be nicer.
        sub_image_path = args.result_prefix + \
                '_{}_at_iteration_{}_{}.png'.format(indx, args.num_scales-1, args.num_iterations_per_scale-1)

        sub_image = img_utils.load_image(sub_image_path)
        shape_before = sub_image.shape[:2]
        sub_image = imresize(sub_image, (height, right-left), interp='bicubic')
        shape_after = sub_image.shape[:2]
        if shape_before != shape_after:
            print('Output image resized from {} to {}'.format(shape_before, shape_after))

        if indx > 0:
            r1 -= TRIMMING
            # The overlapping parts are blended:
            full_output_image[:, l2:r1, :] = blend(full_output_image[:, l2:r1, :], sub_image[:, :r1-l2])
            # The rest simply copied:
            full_output_image[:, r1:r2, :] = sub_image[:, r1-l2:r2-l2, :]
        else:
            full_output_image[:, left:right, :] = sub_image

        previous_sub_image = sub_image

    full_output_image_path = args.result_prefix + '.png'
    imsave(full_output_image_path, full_output_image)
    print('Saving final output {}'.format(full_output_image_path))
