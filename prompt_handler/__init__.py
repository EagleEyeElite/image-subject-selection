def get_all_methods():
    from .max_value import MaxValue
    from .adaptive_dbscan import AdaptiveDBSCAN
    from .adaptive_kmeans import AdaptiveKMeans
    from .random_sampling import RandomSamplingWithImportance
    from .thresholding import ThresholdSaliencyPoints
    from .blob_detection import BlobDetection
    from .contour_detection import ContourDetection
    from .morph_erosion import MorphologicalErosion
    from .negative_grid import NegativeGrid
    from .thresholding import ThresholdSaliencyPoints

    from .negative_grid_max_value import NegativeGridMaxValue
    from .negative_grid_adaptive_kmeans import NegativeGridKMeans
    from .negative_grid_random_sampling import NegativeGridRandomSamplingWithImportance
    from .negative_grid_contour_detection import NegativeGridContourDetection
    from .negative_grid_thresholding import NegativeGridThresholdSaliencyPoints

    return [NegativeGridMaxValue, NegativeGridKMeans, NegativeGridRandomSamplingWithImportance, NegativeGridContourDetection, NegativeGridThresholdSaliencyPoints, MaxValue, AdaptiveKMeans, RandomSamplingWithImportance, ContourDetection, NegativeGrid, ThresholdSaliencyPoints]
    # return [MaxValue, AdaptiveKMeans, RandomSamplingWithImportance, ContourDetection, NegativeGrid, ThresholdSaliencyPoints]


def get_best_methods():
    from .max_value import MaxValue
    from .adaptive_kmeans import AdaptiveKMeans
    from .random_sampling import RandomSamplingWithImportance

    return [MaxValue, AdaptiveKMeans, RandomSamplingWithImportance]
