import tensorflow as tf
from test.dataloader import DetectionInputProcessor, pad_to_fixed_size

MAX_NUM_INSTANCES = 100


def _dataset_parser(value):
    """Parse data to a fixed dimension input image and learning targets.

    Args:
      value: A dictionary contains an image and groundtruth annotations.

    Returns:
      image: Image tensor that is preprocessed to have normalized value and
        fixed dimension [image_size, image_size, 3]
      cls_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      num_positives: Number of positive anchors in the image.
      source_id: Source image id. Default value -1 if the source id is empty
        in the groundtruth annotation.
      image_scale: Scale of the processed image to the original image.
      boxes: Groundtruth bounding box annotations. The box is represented in
        [y1, x1, y2, x2] format. The tensor is padded with -1 to the fixed
        dimension [self._max_num_instances, 4].
      is_crowds: Groundtruth annotations to indicate if an annotation
        represents a group of instances by value {0, 1}. The tensor is
        padded with 0 to the fixed dimension [self._max_num_instances].
      areas: Groundtruth areas annotations. The tensor is padded with -1
        to the fixed dimension [self._max_num_instances].
      classes: Groundtruth classes annotations. The tensor is padded with -1
        to the fixed dimension [self._max_num_instances].
    """
    data = example_decoder.decode(value)
    source_id = data['source_id']
    image = data['image']
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
    areas = data['groundtruth_area']
    is_crowds = data['groundtruth_is_crowd']
    classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])

    input_processor = DetectionInputProcessor(
        image, params['image_size'], boxes, classes)
    input_processor.normalize_image()
    input_processor.set_scale_factors_to_output_size()
    image = input_processor.resize_and_crop_image()
    boxes, classes = input_processor.resize_and_crop_boxes()

    # Assign anchors.
    (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(boxes, classes)

    source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                         source_id)
    source_id = tf.string_to_number(source_id)

    # Pad groundtruth data for evaluation.
    image_scale = input_processor.image_scale_to_original
    boxes *= image_scale
    is_crowds = tf.cast(is_crowds, dtype=tf.float32)
    boxes = pad_to_fixed_size(boxes, -1, [MAX_NUM_INSTANCES, 4])
    is_crowds = pad_to_fixed_size(is_crowds, 0,
                                  [MAX_NUM_INSTANCES, 1])
    areas = pad_to_fixed_size(areas, -1, [MAX_NUM_INSTANCES, 1])
    classes = pad_to_fixed_size(classes, -1, [MAX_NUM_INSTANCES, 1])
    if params['use_bfloat16']:
        image = tf.cast(image, dtype=tf.bfloat16)
    return (image, cls_targets, box_targets, num_positives, source_id,
            image_scale, boxes, is_crowds, areas, classes)


def _get_source_id_from_encoded_image(parsed_tensors):
    return tf.strings.as_string(
        tf.strings.to_hash_bucket_fast(parsed_tensors['image/encoded'],
                                       2 ** 63 - 1))


class TfExampleDecoder(object):
    """Tensorflow Example proto decoder."""

    def __init__(self, include_mask=False, regenerate_source_id=False):
        self._include_mask = include_mask
        self._regenerate_source_id = regenerate_source_id
        self._keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/source_id': tf.io.FixedLenFeature((), tf.string, ''),
            'image/height': tf.io.FixedLenFeature((), tf.int64, -1),
            'image/width': tf.io.FixedLenFeature((), tf.int64, -1),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/area': tf.io.VarLenFeature(tf.float32),
            'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
        }
        if include_mask:
            self._keys_to_features.update({
                'image/object/mask':
                    tf.io.VarLenFeature(tf.string),
            })

    def _decode_image(self, parsed_tensors):
        """Decodes the image and set its static shape."""
        image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def _decode_masks(self, parsed_tensors):
        """Decode a set of PNG masks to the tf.float32 tensors."""

        def _decode_png_mask(png_bytes):
            mask = tf.squeeze(
                tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
            mask = tf.cast(mask, dtype=tf.float32)
            mask.set_shape([None, None])
            return mask

        height = parsed_tensors['image/height']
        width = parsed_tensors['image/width']
        masks = parsed_tensors['image/object/mask']
        return tf.cond(
            tf.greater(tf.size(masks), 0),
            lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
            lambda: tf.zeros([0, height, width], dtype=tf.float32))

    def _decode_areas(self, parsed_tensors):
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.cond(
            tf.greater(tf.shape(parsed_tensors['image/object/area'])[0], 0),
            lambda: parsed_tensors['image/object/area'],
            lambda: (xmax - xmin) * (ymax - ymin))

    def decode(self, serialized_example):
        """Decode the serialized example.

        Args:
          serialized_example: a single serialized tf.Example string.

        Returns:
          decoded_tensors: a dictionary of tensors with the following fields:
            - image: a uint8 tensor of shape [None, None, 3].
            - source_id: a string scalar tensor.
            - height: an integer scalar tensor.
            - width: an integer scalar tensor.
            - groundtruth_classes: a int64 tensor of shape [None].
            - groundtruth_is_crowd: a bool tensor of shape [None].
            - groundtruth_area: a float32 tensor of shape [None].
            - groundtruth_boxes: a float32 tensor of shape [None, 4].
            - groundtruth_instance_masks: a float32 tensor of shape
                [None, None, None].
            - groundtruth_instance_masks_png: a string tensor of shape [None].
        """
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, self._keys_to_features)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value='')
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0)

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        areas = self._decode_areas(parsed_tensors)

        decode_image_shape = tf.logical_or(
            tf.equal(parsed_tensors['image/height'], -1),
            tf.equal(parsed_tensors['image/width'], -1))
        image_shape = tf.cast(tf.shape(image), dtype=tf.int64)

        parsed_tensors['image/height'] = tf.where(decode_image_shape,
                                                  image_shape[0],
                                                  parsed_tensors['image/height'])
        parsed_tensors['image/width'] = tf.where(decode_image_shape, image_shape[1],
                                                 parsed_tensors['image/width'])

        is_crowds = tf.cond(
            tf.greater(tf.shape(parsed_tensors['image/object/is_crowd'])[0], 0),
            lambda: tf.cast(parsed_tensors['image/object/is_crowd'], dtype=tf.bool),
            lambda: tf.zeros_like(parsed_tensors['image/object/class/label'],
                                  dtype=tf.bool))  # pylint: disable=line-too-long
        if self._regenerate_source_id:
            source_id = _get_source_id_from_encoded_image(parsed_tensors)
        else:
            source_id = tf.cond(
                tf.greater(tf.strings.length(parsed_tensors['image/source_id']),
                           0), lambda: parsed_tensors['image/source_id'],
                lambda: _get_source_id_from_encoded_image(parsed_tensors))
        if self._include_mask:
            masks = self._decode_masks(parsed_tensors)

        decoded_tensors = {
            'image': image,
            'source_id': source_id,
            'height': parsed_tensors['image/height'],
            'width': parsed_tensors['image/width'],
            'groundtruth_classes': parsed_tensors['image/object/class/label'],
            'groundtruth_is_crowd': is_crowds,
            'groundtruth_area': areas,
            'groundtruth_boxes': boxes,
        }
        if self._include_mask:
            decoded_tensors.update({
                'groundtruth_instance_masks': masks,
                'groundtruth_instance_masks_png': parsed_tensors['image/object/mask'],
            })
        return decoded_tensors


if __name__ == '__main__':
    import json
    from test import anchors

    dataset = tf.data.TFRecordDataset(['datasets/coco/coco_train.record-00090-of-00100'])
    example_decoder = TfExampleDecoder()
    params = json.load(open('test/params.json'))
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    params['image_size'])
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
    for raw_record in dataset.take(1):
        # decoded_tensors = example_decoder.decode(raw_record)
        _dataset_parser(raw_record)
    # dataset.map(_dataset_parser)
