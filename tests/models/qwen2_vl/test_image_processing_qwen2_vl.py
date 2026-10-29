# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import tempfile
import unittest

import numpy as np

from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import (
    ImageProcessingTester,
    ImageProcessingTestMixin,
    load_coco_image,
    prepare_image_inputs,
    prepare_video_inputs,
)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class Qwen2VLImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        # Random test inputs kwargs
        kwargs.setdefault("num_frames", 10)
        kwargs.setdefault("min_resolution", 56)
        kwargs.setdefault("max_resolution", 1024)

        # Image processor init kwargs
        kwargs.setdefault("min_pixels", 56 * 56)
        kwargs.setdefault("max_pixels", 28 * 28 * 1280)

        super().__init__(**kwargs)

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return [[image] for image in images]

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            num_frames=self.num_frames,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class Qwen2VLImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = Qwen2VLImageProcessingTester

    def test_image_processor_to_json_string(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            obj = json.loads(image_processor.to_json_string())
            for key, value in self.image_processor_dict.items():
                if key not in ["min_pixels", "max_pixels"]:
                    self.assertEqual(obj[key], value)

    def test_class_size_mutation(self):
        """
        Passing a deprecated min/max pixels should NOT mutate cls.size. Pssing
        only one of the min or max pixels, should not raise any errors and keep
        the default `cls.size` for missing entries.
        """
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)

        for image_processing_class in self.image_processing_classes.values():
            original_size = dict(image_processing_class().size)

            # Instantiate with min_pixels to ensure it doesn't mutate class defaults
            processor = image_processing_class(min_pixels=123)
            self.assertEqual(processor.size.get("shortest_edge"), 123)

            # The default is already `123`, so we get identical pixels
            encoded_images_1 = processor(image_inputs, min_pixels=123, return_tensors="pt")
            encoded_images_2 = processor(image_inputs, return_tensors="pt")
            torch.allclose(encoded_images_1.pixel_values, encoded_images_2.pixel_values)

            # Check if class default was mutated
            processor_reloaded = image_processing_class()
            self.assertEqual(dict(processor_reloaded.size), original_size)

    def test_select_best_resolution(self):
        # Test with a final resize resolution
        best_resolution = smart_resize(561, 278, factor=28)
        self.assertEqual(best_resolution, (560, 280))

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], Image.Image)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], np.ndarray)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image[0], torch.Tensor)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    @unittest.skip(reason="Qwen2VLImageProcessor doesn't treat 4 channel PIL and numpy consistently yet")
    def test_call_numpy_4_channels(self):
        pass

    def test_nested_input(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # Test batched as a list of images
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched as a nested list of images, where each sublist is one batch
            image_inputs_nested = image_inputs[:3] + image_inputs[3:]
            process_out = image_processing(image_inputs_nested, return_tensors="pt")
            encoded_images_nested = process_out.pixel_values
            image_grid_thws_nested = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Image processor should return same pixel values, independently of ipnut format
            self.assertTrue((encoded_images_nested == encoded_images).all())
            self.assertTrue((image_grid_thws_nested == expected_image_grid_thws).all())

    def test_custom_image_size(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                image_processing.save_pretrained(tmpdirname)
                image_processor_loaded = image_processing_class.from_pretrained(
                    tmpdirname, max_pixels=56 * 56, min_pixels=28 * 28
                )

            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            process_out = image_processor_loaded(image_inputs, return_tensors="pt")
            expected_output_video_shape = [112, 1176]
            self.assertListEqual(list(process_out.pixel_values.shape), expected_output_video_shape)

    def test_custom_pixels(self):
        pixel_choices = frozenset(itertools.product((100, 150, 200, 20000), (100, 150, 200, 20000)))
        for image_processing_class in self.image_processing_classes.values():
            image_processor_dict = self.image_processor_dict.copy()
            for a_pixels, b_pixels in pixel_choices:
                image_processor_dict["min_pixels"] = min(a_pixels, b_pixels)
                image_processor_dict["max_pixels"] = max(a_pixels, b_pixels)
                image_processor = image_processing_class(**image_processor_dict)
                image_inputs = self.image_processor_tester.prepare_image_inputs()
                # Just checking that it doesn't raise an error
                image_processor(image_inputs, return_tensors="pt")

    @require_vision
    @require_torch
    def test_backends_equivalence(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = load_coco_image("000000039769.jpg")

        # Create processors for each backend
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding.pixel_values, encodings[backend_name].pixel_values)
            self.assertEqual(reference_encoding.image_grid_thw.dtype, encodings[backend_name].image_grid_thw.dtype)
            self._assert_tensors_equivalence(
                reference_encoding.image_grid_thw.float(), encodings[backend_name].image_grid_thw.float()
            )

    @require_vision
    @require_torch
    def test_backends_equivalence_batched(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        # Create processors for each backend
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding.pixel_values, encodings[backend_name].pixel_values)
            self.assertEqual(reference_encoding.image_grid_thw.dtype, encodings[backend_name].image_grid_thw.dtype)
            self._assert_tensors_equivalence(
                reference_encoding.image_grid_thw.float(), encodings[backend_name].image_grid_thw.float()
            )

    def test_get_num_patches_without_images(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            num_patches = image_processing.get_number_of_image_patches(height=100, width=100, images_kwargs={})
            self.assertEqual(num_patches, 64)

            num_patches = image_processing.get_number_of_image_patches(height=200, width=50, images_kwargs={})
            self.assertEqual(num_patches, 56)

            num_patches = image_processing.get_number_of_image_patches(
                height=100, width=100, images_kwargs={"patch_size": 28}
            )
            self.assertEqual(num_patches, 16)

    def test_batch_patchification_matches_individual_images(self):
        """A shared patch buffer must preserve image order, values, dtype, and input ownership."""
        cls = self.image_processing_classes["torchvision"]
        for dtype, grouping, patch_config, transforms in itertools.product(
            (torch.uint8, torch.float32),
            (None, True, False),
            ((14, 2, 2), (16, 1, 1)),
            itertools.product((False, True), repeat=2),
        ):
            with self.subTest(dtype=dtype, grouping=grouping, patches=patch_config, transforms=transforms):
                patch_size, merge_size, temporal_patch_size = patch_config
                do_rescale, do_normalize = transforms
                processor = cls(
                    do_resize=False,
                    do_rescale=do_rescale,
                    do_normalize=do_normalize,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    temporal_patch_size=temporal_patch_size,
                )
                factor = patch_size * merge_size
                shapes = [(2 * factor, 3 * factor), (3 * factor, 2 * factor), (2 * factor, 3 * factor)]
                generator = torch.Generator().manual_seed(0)
                images = [
                    torch.randint(0, 256, (3, h, w), generator=generator).to(device=torch_device, dtype=dtype)
                    for h, w in shapes
                ]
                originals = [image.clone() for image in images]
                expected = [processor(image, return_tensors="pt") for image in images]

                actual = processor(images, return_tensors="pt", disable_grouping=grouping)

                torch.testing.assert_close(
                    actual.pixel_values, torch.cat([item.pixel_values for item in expected]), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual.image_grid_thw, torch.cat([item.image_grid_thw for item in expected])
                )
                for image, original in zip(images, originals):
                    torch.testing.assert_close(image, original, rtol=0, atol=0)

    def test_single_group_patchification_reuses_patch_storage(self):
        """A single shape group must reuse patch storage while retaining its gradients."""
        cls = self.image_processing_classes["torchvision"]

        class CapturePatchProcessor(cls):
            def patchify(self, images, patch_size, merge_size, temporal_patch_size):
                patches, grid_h, grid_w = super().patchify(images, patch_size, merge_size, temporal_patch_size)
                self.patches = patches
                return patches, grid_h, grid_w

        for batch_size, grouping in ((1, None), (1, True), (1, False), (3, False)):
            with self.subTest(batch_size=batch_size, grouping=grouping):
                processor = CapturePatchProcessor(do_resize=False, do_rescale=False, do_normalize=False)
                images = torch.randn(batch_size, 3, 56, 84, device=torch_device, requires_grad=True)
                image_inputs = images[0] if batch_size == 1 else images
                output = processor(image_inputs, return_tensors="pt", disable_grouping=grouping).pixel_values

                self.assertEqual(output.data_ptr(), processor.patches.data_ptr())
                torch.testing.assert_close(output, processor.patches.flatten(0, 1), rtol=0, atol=0)
                gradient = torch.autograd.grad(output.square().sum(), images)[0]
                torch.testing.assert_close(gradient, 2 * processor.temporal_patch_size * images)

    def test_batch_patchification_preserves_subclass_override(self):
        """Batch processing must continue to honor the public patchification hook."""
        cls = self.image_processing_classes["torchvision"]

        class CustomPatchProcessor(cls):
            def patchify(self, images, patch_size, merge_size, temporal_patch_size):
                patches, grid_h, grid_w = super().patchify(images, patch_size, merge_size, temporal_patch_size)
                return patches.flip(1), grid_h, grid_w

        processor = CustomPatchProcessor(do_resize=False)
        images = [torch.arange(3 * 56 * 56, dtype=torch.float32).reshape(3, 56, 56)] * 2
        expected = torch.cat([processor(image, return_tensors="pt").pixel_values for image in images])

        actual = processor(images, return_tensors="pt")

        torch.testing.assert_close(actual.pixel_values, expected, rtol=0, atol=0)

    def test_grouped_patchification_preserves_gradients(self):
        """Interleaved shape groups must preserve patch values and gradients."""
        cls = self.image_processing_classes["torchvision"]
        for do_normalize, do_rescale in itertools.product((False, True), repeat=2):
            with self.subTest(do_normalize=do_normalize, do_rescale=do_rescale):
                processor = cls(do_resize=False, do_rescale=do_rescale, do_normalize=do_normalize)
                generator = torch.Generator().manual_seed(0)
                images = [
                    torch.randn(shape, dtype=dtype, generator=generator).to(torch_device).requires_grad_()
                    for shape, dtype in (
                        ((3, 56, 84), torch.float32),
                        ((3, 84, 56), torch.float32),
                        ((3, 56, 84), torch.float32),
                    )
                ]
                expected_patches = []
                for image in images:
                    normalized = processor.rescale_and_normalize(
                        image.unsqueeze(0),
                        do_rescale,
                        processor.rescale_factor,
                        do_normalize,
                        processor.image_mean,
                        processor.image_std,
                    )
                    patches, _, _ = processor.patchify(
                        normalized, processor.patch_size, processor.merge_size, processor.temporal_patch_size
                    )
                    expected_patches.append(patches[0])
                expected = torch.cat(expected_patches)

                actual = processor(images, return_tensors="pt", disable_grouping=False).pixel_values

                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual_grads = torch.autograd.grad(actual.square().sum(), images)
                expected_grads = torch.autograd.grad(expected.square().sum(), images)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)
