import torch
import torchvision.transforms.functional as F
import warnings
import random
import numpy as np
import torchvision
from PIL import Image, ImageOps, ImageFilter
import numbers
import PIL


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.
    Args:
        img (PIL Image): Image to be converted to grayscale.
    Returns:
        PIL Image: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not isinstance(img,PIL.Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img



class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return (out_images, label)


class GroupRandomGrayScale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """
    def __init__(self, p=0.2, per_frame=False):
        super().__init__()
        self.p = p
        self.per_frame = per_frame
    def __call__(self, img_tuple):
        """
        Args:
            list of imgs (PIL Image or Tensor): Image to be converted to grayscale.
        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        clip, label = img_tuple
        num_output_channels = 1 if clip[0].mode == 'L' else 3
        if self.per_frame:
            for i in range(len(clip)):
                if random.random() < self.p:
                    clip[i]=to_grayscale(clip[i],num_output_channels)
        else:
            if torch.rand(1)<self.p:
                for i in range(len(clip)):
                    clip[i]=to_grayscale(clip[i],num_output_channels)
        return (clip, label)


class GroupRandomGaussianBlur(object):
    """Apply gaussian blur on a list of images
    Args:
    p (float): probability of applying the transformation
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2., per_frame=False):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.per_frame = per_frame

    def __call__(self, img_tuple):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be blurred
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Blurred list of images
        """
        clip, label = img_tuple
        if self.per_frame:
            for i in range(len(clip)):
                if random.random() < self.p:
                    radius = random.uniform(self.radius_min, self.radius_max)
                    if isinstance(clip[0], np.ndarray):
                        clip[i] = skimage.filters.gaussian(clip[i])
                    elif isinstance(clip[0], PIL.Image.Image):
                        clip[i] = clip[i].filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
                    else:
                        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                        'but got list of {0}'.format(type(clip[0])))
            return (clip, label)
        else:
            if random.random() < self.p:
                if isinstance(clip[0], np.ndarray):
                    blurred = [skimage.filters.gaussian(img) for img in clip]
                elif isinstance(clip[0], PIL.Image.Image):
                    blurred = [img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))) for img in clip]
                else:
                    raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                    'but got list of {0}'.format(type(clip[0])))
                return (blurred, label)
            else:
                return (clip, label)



class GroupColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, per_frame=False):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.per_frame = per_frame

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, img_tuple):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        clip, label = img_tuple
        jittered_clip = []
        if self.per_frame:
            for img in clip:
                if isinstance(clip[0], np.ndarray):
                    raise TypeError(
                        'Color jitter not yet implemented for numpy arrays')
                elif isinstance(clip[0], PIL.Image.Image):
                    brightness, contrast, saturation, hue = self.get_params(
                        self.brightness, self.contrast, self.saturation, self.hue)
                    # Create img transform function sequence
                    img_transforms = []
                    if brightness is not None:
                        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
                    if saturation is not None:
                        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
                    if hue is not None:
                        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
                    if contrast is not None:
                        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
                    random.shuffle(img_transforms)
                    for func in img_transforms:
                        jittered_img = func(img)
                    jittered_clip.append(jittered_img)
                
                else:
                    raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                    'but got list of {0}'.format(type(clip[0])))

        else:
            if isinstance(clip[0], np.ndarray):
                raise TypeError(
                    'Color jitter not yet implemented for numpy arrays')
            elif isinstance(clip[0], PIL.Image.Image):
                brightness, contrast, saturation, hue = self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

                # Create img transform function sequence
                img_transforms = []
                if brightness is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
                if saturation is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
                if hue is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
                if contrast is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
                random.shuffle(img_transforms)

                # Apply to all images
                for img in clip:
                    for func in img_transforms:
                        jittered_img = func(img)
                    jittered_clip.append(jittered_img)

            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(clip[0])))
        return (jittered_clip, label)


class GroupRandomApply(object):
    """Apply a list of transformations with a probability p
    Args:
    transforms (list of Transform objects): list of transformations to compose.
    p (float): probability of applying the transformations
    """

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tuple):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be transformed
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Transformed list of images
        """
        clip, label = img_tuple  
        if random.random() < self.p:
            for t in self.transforms:
                (clip, label) = t((clip, label))
        return (clip, label)


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_tuple):
        tensor, label = tensor_tuple
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))
        
        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return (tensor,label)


class GroupGrayScale(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.Grayscale(size)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)

    
class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)

class GroupResize(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.Resize((size,size)) # default bilinear

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        out_images = [self.worker(img) for img in img_group]
        return (out_images, label)
    
class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]
        return (ret_img_group, label)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        if img_group[0].mode == 'L':
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate(img_group, axis=2), label)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple
        
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return (img.float().div(255.) if self.div else img.float(), label)


class IdentityTransform(object):

    def __call__(self, data):
        return data
