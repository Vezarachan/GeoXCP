from typing import Callable
import geopandas as gpd
import numpy as np
import pandas as pd


class GeoConformalSpatialPrediction:
    def __init__(self, predict_f: Callable, nonconformity_score_f: Callable = None,
                 miscoverage_level: float = 0.1, bandwidth: float = None,
                 coord_calib: np.ndarray = None, coord_test: np.ndarray = None,
                 X_calib: np.ndarray = None, y_calib: np.ndarray = None,
                 X_test: np.ndarray = None, y_test: np.ndarray = None):
        """
        Initialize the GeoConformalSpatialPrediction

        predict_f: the predict function of spatial prediction model
        nonconformity_score_f: a measure of how well a new data point conforms to a model trained on a given dataset
        miscoverage_level: the percentage of data points not in the confidence interval
        bandwidth: the bandwidth of the Gaussian kernel
        test_data: data points used for measuring uncertainty
        coord_calib: coordinates of calibration data points
        coord_test: coordinates of test data points
        """
        self.predict_f = predict_f
        self.nonconformity_score_f = nonconformity_score_f
        self.miscoverage_level = miscoverage_level
        self.bandwidth = bandwidth
        self.coord_calib = coord_calib
        self.coord_test = coord_test
        self.X_calib = X_calib
        self.y_calib = y_calib
        self.X_test = X_test
        self.y_test = y_test
        self.uncertainty = None
        self.geo_uncertainty = None
        self.geo = None
        self.upper_bound = None
        self.lower_bound = None
        self.predicted_value = None
        self.coverage_proba = None

    def predict_geoconformal_uncertainty(self):
        """
        Calculate the geographically weighted uncertainty for each sample
        """
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._default_nonconformity_score
        y_calib_pred = self.predict_f(self.X_calib)
        nonconformity_scores = np.array(self.nonconformity_score_f(y_calib_pred, self.y_calib))
        uncertainty_list = []
        for p in self.coord_test:
            weights = self._kernel_smoothing(p, self.coord_calib, self.bandwidth)
            quantile = self._weighted_quantile(nonconformity_scores, self.miscoverage_level, weights)
            uncertainty_list.append(quantile)
        uncertainty = np.quantile(nonconformity_scores, 1 - self.miscoverage_level)
        self.geo_uncertainty = np.array(uncertainty_list)
        self.uncertainty = uncertainty

    def predict_prediction_interval(self):
        """
        Calculate the confidence interval based on uncertainty
        :return:
        """
        predicted_value = self.predict_f(self.X_test)
        upper_bound = predicted_value + self.geo_uncertainty
        lower_bound = predicted_value - self.geo_uncertainty
        self.predicted_value = predicted_value
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def coverage_probability(self) -> float:
        """
        Calculate the coverage probability of confidence interval
        """
        self.coverage_proba = np.mean((self.y_test >= self.lower_bound) & (self.y_test <= self.upper_bound))

    def analyze(self):
        self.predict_geoconformal_uncertainty()
        self.predict_prediction_interval()
        self.coverage_probability()
        return GeoConformalResults(self.geo_uncertainty, self.uncertainty, self.coord_test, self.predicted_value,
                                   self.upper_bound, self.lower_bound, self.coverage_proba)

    def _default_nonconformity_score(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Default equation for computing nonconformity score
        :param pred: predicted values
        :param gt: ground truth values
        :return: list of nonconformity scores
        """
        return np.abs(pred - gt)

    def _gaussian_kernel(self, d: np.ndarray) -> np.ndarray:
        """
        Gaussian distance decay function
        :param d: distances from test samples to calibration samples
        :return: list of weights for calibration samples
        """
        return np.exp(-0.5 * d ** 2)

    def _kernel_smoothing(self, z_test: np.ndarray, z_calib: np.ndarray, bandwidth: float) -> np.ndarray:
        """
        Kernel smoothing function
        :param z_test: the coordinates of test samples
        :param z_calib: the coordinates of calibration samples
        :param bandwidth: distance decay parameter
        :return: list of weights for calibration samples
        """
        distances = np.sqrt(np.sum(np.square(z_calib - z_test), axis=1))
        # distances = np.abs(z_calib - z_test)
        weights = self._gaussian_kernel(distances / bandwidth)
        return weights

    def _weighted_quantile(self, scores: np.ndarray, alpha: float = 0.1, weights: np.ndarray = None):
        """
        Calculate weighted quantile
        :param scores: nonconformity scores
        :param alpha: miscoverage level
        :param weights: geographic weights
        :return: weighted quantile at (1-alpha) miscoverage level
        """
        if weights is None:
            weights = np.ones(len(scores))
        # print(weights.shape)
        # print(scores.shape)

        sorted_indices = np.argsort(scores)
        scores_sorted = scores[sorted_indices]
        weights_sorted = weights[sorted_indices]

        cumsum_weights = np.cumsum(weights_sorted)
        normalized_cumsum_weights = cumsum_weights / cumsum_weights[-1]

        idx = np.searchsorted(normalized_cumsum_weights, 1 - alpha)
        return scores_sorted[idx]


class GeoConformalResults:
    def __init__(self, geo_uncertainty: np.ndarray, uncertainty: float, coords: np.ndarray, pred: np.ndarray,
                 upper_bound: np.ndarray, lower_bound: np.ndarray, coverage_probability: float,
                 crs: str = 'EPSG:4326'):
        self.uncertainty = uncertainty
        self.geo_uncertainty = geo_uncertainty
        self.coords = coords
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.pred = pred
        self.coverage_probability = coverage_probability
        self.crs = crs

    def to_gpd(self) -> gpd.GeoDataFrame:
        geo_uncertainty_pd = pd.DataFrame(np.column_stack(
            [self.geo_uncertainty, self.pred, self.upper_bound, self.lower_bound, self.coords]))
        geo_uncertainty_pd.columns = ['geo_uncertainty', 'pred', 'upper_bound', 'lower_bound', 'x', 'y']
        geo_uncertainty_gpd = gpd.GeoDataFrame(geo_uncertainty_pd, crs=self.crs,
                                               geometry=gpd.points_from_xy(x=geo_uncertainty_pd.x,
                                                                           y=geo_uncertainty_pd.y))
        return geo_uncertainty_gpd
