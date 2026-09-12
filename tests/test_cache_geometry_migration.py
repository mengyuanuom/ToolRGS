import unittest
import warnings
import numpy as np
from toolrgs.evaluation.prediction_cache import score_prediction_cache
from utils.grasp_raster import GEOMETRY_VERSION


def old_cache():
    return dict(metadata=dict(max_topk=5),segmentation_iou=np.array([1.]),
                rectangle_offsets=np.array([0,1]),target_offsets=np.array([0,1]),
                match_offsets=np.array([0,1]),rectangles=np.array([[1000.,360.,300.,20.,0.]]),
                targets=np.array([[1000.,360.,300.,20.,0.,0.]]),
                matches=np.array([[0.,0.,0.]]),target_width_cap=np.array([300.]),
                target_height=np.array([20.]))


class CacheMigrationTest(unittest.TestCase):
    def test_old_iou_not_reused(self):
        cache=old_cache()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s=score_prediction_cache(cache,image_shape=(720,1280))
        self.assertEqual(s['j_index'],[1.,1.])
        self.assertEqual(s['msr'][1],1.)
        self.assertEqual(cache['metadata']['geometry_version'],GEOMETRY_VERSION)
        np.testing.assert_equal(cache['rectangles'][0],[1000.,360.,300.,20.,0.])

    def test_old_shape_required(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(ValueError):score_prediction_cache(old_cache())

    def test_versioned_cache_scores_without_shape_override(self):
        cache=old_cache()
        cache['image_hw']=np.array([[720,1280]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a=score_prediction_cache(cache)
        b=score_prediction_cache(cache)
        self.assertEqual(a['j_index'],b['j_index'])


if __name__=='__main__':unittest.main()
