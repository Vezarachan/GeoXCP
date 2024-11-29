from typing import Callable, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from GeoConformal import GeoConformalSpatialPrediction
from GeoConformal.geocp import GeoConformalResults
import geoplot as gplt
import geoplot.crs as gcrs
from joblib import Parallel, delayed
from tqdm import tqdm


class GeoConformalizedExplainer:
    def __init__(self, model: Any, x_train: np.ndarray, x_calib: np.ndarray, coord_calib: np.ndarray = None,
                 coord_test: np.ndarray = None, miscoverage_level: float = 0.1, band_width: float = None):
        self.model = model
        self.x_train = x_train
        self.x_calib = x_calib
        _, k = x_calib.shape
        self.num_variables = k
        self.coord_calib = coord_calib
        self.coord_test = coord_test
        self.miscoverage_level = miscoverage_level
        self.band_width = band_width

    def _compute_explanation_values(self, x: np.ndarray) -> shap.Explanation:
        """
        Compute explanation values with explanation methods such as SHAP. LIME, etc.
        :param x:
        :return:
        """
        explainer = shap.Explainer(model=self.model)
        explanation_values = explainer(x)
        return explanation_values

    def _fit_explanation_value_predictor(self, x: np.ndarray, t: np.ndarray, s: np.ndarray) -> XGBRegressor:
        """
        Fit the regression model between explanation values of a variable and input values X, predicted values t.
        :param X:
        :param t: prediction values by the black-box model
        :param s: ground truth explanation values
        :return:
        """
        params = {'eta': [0.01, 0.1, 1.0], 'gamma': [0, 0.1],
                  'n_estimators': [10, 100, 500], 'max_depth': [2, 4, 6],
                  'min_child_weight': [1, 2], 'nthread': [2]}
        kf = KFold(n_splits=3, random_state=1, shuffle=True)
        model = XGBRegressor()
        model_cv = GridSearchCV(estimator=model,
                                 param_grid=params,
                                 verbose=0,
                                 return_train_score=False,
                                 n_jobs=1,
                                 cv=kf)
        t = t.reshape(-1, 1)
        x_new = np.hstack((x, t))
        model_cv.fit(x_new, s)
        regressor = XGBRegressor(params=model_cv.best_params_)
        regressor.fit(x_new, s)
        return regressor

    def _predict_explanation_values(self, x: np.ndarray, t: np.ndarray, regression_predict_f: Callable) -> np.ndarray:
        """
        Predict explanation values from input values X and predicted values t.
        :param X:
        :param t:
        :return:
        """
        t = t.reshape(-1, 1)
        x_new = np.hstack((x, t))
        return regression_predict_f(x_new)

    def _explain_ith_variable(self, i: int, s_train: np.ndarray, t_train: np.ndarray, x_test_new: np.ndarray, s_test: np.ndarray, x_calib_new: np.ndarray, s_calib: np.ndarray):
        regressor = self._fit_explanation_value_predictor(self.x_train, t_train, s_train[:, i])
        r2 = regressor.score(x_test_new, s_test[:, i])
        rmse = root_mean_squared_error(regressor.predict(x_test_new), s_test[:, i])
        geocp = GeoConformalSpatialPrediction(predict_f=regressor.predict,
                                              miscoverage_level=self.miscoverage_level,
                                              bandwidth=self.band_width,
                                              coord_calib=self.coord_calib,
                                              coord_test=self.coord_test,
                                              X_calib=x_calib_new, y_calib=s_calib[:, i],
                                              X_test=x_test_new, y_test=s_test[:, i])
        result_ith_variable = geocp.analyze()
        return result_ith_variable, r2, rmse

    def uncertainty_aware_explain(self, x_test: pd.DataFrame, n_jobs: int = 4, is_geo: bool = False) -> shap.Explanation:
        t_train = self.model.predict(self.x_train)
        shap_train = self._compute_explanation_values(self.x_train)
        s_train = shap_train.values
        t_calib = self.model.predict(self.x_calib).reshape(-1, 1)
        shap_calib = self._compute_explanation_values(self.x_calib)
        s_calib = shap_calib.values
        x_calib_new = np.hstack((self.x_calib, t_calib))
        shap_test = self._compute_explanation_values(x_test)
        s_test = shap_test.values
        t_test = self.model.predict(x_test).reshape(-1, 1)
        x_test_new = np.hstack((x_test, t_test))
        # Parallelize the regressor fitting and conformal prediction of all variables
        results = Parallel(n_jobs=n_jobs)(delayed(self._explain_ith_variable)(i, s_train, t_train, x_test_new, s_test, x_calib_new, s_calib) for i in tqdm(range(self.num_variables)))
        geocp_results = [result[0] for result in results]
        r2 = np.array([result[1] for result in results])
        rmse = np.array([result[2] for result in results])
        return GeoConformalizedExplainerResults(explanation=shap_test, geocp_results=geocp_results, regression_scores=r2, regression_rmse=rmse, coords=self.coord_test, feature_values=x_test.values)


class GeoConformalizedExplainerResults:
    def __init__(self, explanation: shap.Explanation, geocp_results: List[GeoConformalResults], regression_scores: np.ndarray, regression_rmse: np.ndarray, coords: np.ndarray, feature_values: np.ndarray, crs: str = 'EPSG:4326'):
        self.explanation_values = explanation.values
        self.feature_names = explanation.feature_names
        self.geocp_results = geocp_results
        self.coords = coords
        self.regression_scores = regression_scores
        self.regression_rmse = regression_rmse
        self.crs = crs
        self.feature_values = feature_values
        self.result = self._get_shap_values_with_uncertainty()

    def _get_shap_values_with_uncertainty(self) -> pd.DataFrame:
        df = pd.DataFrame(self.explanation_values, columns=self.feature_names)
        for i in range(len(self.feature_names)):
            feature_name = self.feature_names[i]
            geocp_result = self.geocp_results[i]
            df[f'{feature_name}_geo_uncertainty'] = geocp_result.geo_uncertainty
            df[f'{feature_name}_uncertainty'] = geocp_result.uncertainty
            df[f'{feature_name}_upper_bound'] = geocp_result.upper_bound
            df[f'{feature_name}_lower_bound'] = geocp_result.lower_bound
            df[f'{feature_name}_coverage_probability'] = geocp_result.coverage_probability
            df[f'{feature_name}_pred'] = geocp_result.pred
        df['x'] = self.coords[:, 0]
        df['y'] = self.coords[:, 1]
        return df

    def plot_absolute_shap_value_with_uncertainty(self):
        plt.rcParams['font.size'] = 12
        mean_abs_importance = np.mean(np.abs(self.explanation_values), axis=0)
        index = np.argsort(mean_abs_importance)
        sorted_mean_abs_importance = mean_abs_importance[index]
        sorted_feature_names = np.array(self.feature_names)[index]
        uncertainty = []
        for i in range(len(self.feature_names)):
            uncertainty.append(self.geocp_results[i].uncertainty)
        sorted_uncertainty = np.array(uncertainty)[index]
        fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 10))
        axes[0].barh(sorted_feature_names, sorted_mean_abs_importance, align='center', color='#ff0d57')
        axes[1].barh(sorted_feature_names, sorted_uncertainty, align='center', color='#1e88e5')
        axes[0].set(title='mean(|SHAP Value|)')
        axes[1].set(title='Uncertainty')
        axes[0].invert_xaxis()
        axes[0].set(yticks=np.arange(len(sorted_feature_names)), yticklabels=sorted_feature_names)
        axes[0].yaxis.tick_right()
        fig.tight_layout()
        plt.show()

    def plot_shap_values_with_uncertainty(self, i: int, filename: str = None):
        plt.rcParams['font.size'] = 12
        fig, ax = plt.subplots(figsize=(10, 10))
        result_i = self.result.iloc[i]
        feature_values_i = self.feature_values[i, :]
        shap_values_i = result_i[self.feature_names]
        lower_bound_list = []
        upper_bound_list = []
        for feature_name in self.feature_names:
            lower_bound_list.append(result_i[f'{feature_name}_lower_bound'])
            upper_bound_list.append(result_i[f'{feature_name}_upper_bound'])
        colors = ['#ff0d57' if e >= 0 else '#1e88e5' for e in shap_values_i]
        labels = [f'+{e:.2f}' if e >= 0 else f'{e:.2f}' for e in shap_values_i]

        bars = ax.barh(self.feature_names, shap_values_i, color=colors)
        y_positions = np.arange(len(self.feature_names))

        for bar, label in zip(bars, labels):
            width = bar.get_width()
            ax.annotate(
                label,
                xy=(width, bar.get_y() + bar.get_height() / 2),  # Position
                xytext=(0, 15),  # Offset (x, y) in points
                textcoords="offset points",  # Relative positioning
                ha='left', va='center',  # Horizontal and vertical alignment
                weight='bold'
            )

        for y, x, low, high in zip(y_positions, shap_values_i, lower_bound_list, upper_bound_list):
            ax.plot([low, high], [y, y], color='#454545', linewidth=1.5, solid_capstyle='butt', zorder=1)
            ax.plot([low, low], [y - 0.06, y + 0.06], color='#454545', linewidth=1.5, solid_capstyle='butt', zorder=1)
            ax.plot([high, high], [y - 0.06, y + 0.06], color='#454545', linewidth=1.5, solid_capstyle='butt', zorder=1)
        plt.xlabel('Importance')

        y_ticks = []
        for name, value in zip(self.feature_names, feature_values_i):
            y_ticks.append(f'{name} = {value}')

        plt.yticks(y_positions, y_ticks)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def to_gdf(self) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(self.result, crs=self.crs,
                                               geometry=gpd.points_from_xy(x=self.result.x,
                                                                           y=self.result.y))
        return gdf

    def accuracy_summary(self) -> pd.DataFrame:
        coverage_proba_list = []
        for name in self.feature_names:
            coverage_name = f'{name}_coverage_probability'
            coverage_proba = self.result[coverage_name][0]
            coverage_proba_list.append(coverage_proba)
        coverage_proba_list = np.array(coverage_proba_list).reshape(-1, 1)
        df = pd.DataFrame(np.hstack((coverage_proba_list, self.regression_scores.reshape(-1, 1), self.regression_rmse.reshape(-1, 1))), columns=['coverage_probability', 'R2', 'RMSE'])
        df.index = self.feature_names
        return df

