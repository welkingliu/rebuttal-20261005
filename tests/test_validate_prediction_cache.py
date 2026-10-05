import unittest

from sgg_core.tools.validate_prediction_cache import _dataset_image_ids


class DatasetImageIdsTest(unittest.TestCase):
    def test_tuple_items_do_not_materialize_samples(self):
        class Dataset:
            items = [("image-a", object()), (22, object())]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, _):
                raise AssertionError("validation must not materialize a sample")

        self.assertEqual(_dataset_image_ids(Dataset()), ["image-a", "22"])

    def test_vg_indices_use_external_image_ids(self):
        class Dataset:
            image_indices = [3, 7]
            index_to_image_meta = {3: {"image_id": 103}, 7: {"image_id": 107}}

            def __len__(self):
                return 2

        self.assertEqual(_dataset_image_ids(Dataset()), ["103", "107"])

    def test_unknown_layout_falls_back(self):
        class Dataset:
            def __len__(self):
                return 1

        self.assertIsNone(_dataset_image_ids(Dataset()))


if __name__ == "__main__":
    unittest.main()
