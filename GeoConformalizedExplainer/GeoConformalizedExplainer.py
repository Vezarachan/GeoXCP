from typing import Callable, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from joblib import Parallel, delayed
from tqdm import tqdm
import geoplot as gplt
import geoplot.crs as gcrs
import contextily as cx
from math import ceil
from pygam import LinearGAM, s
from GeoConformal import GeoConformalSpatialPrediction
from GeoConformal.geocp import GeoConformalResults


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
                                 n_jobs=4,
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
        self.explanation = explanation
        self.explanation_values = explanation.values
        self.feature_names = explanation.feature_names
        self.geocp_results = geocp_results
        self.coords = coords
        self.regression_scores = regression_scores
        self.regression_rmse = regression_rmse
        self.crs = crs
        self.feature_values = feature_values
        self.result = self._get_shap_values_with_uncertainty()
        self.result_geo = self.to_gdf()

    def _get_shap_values_with_uncertainty(self) -> pd.DataFrame:
        feature_shap_names = list(map(lambda s: f'{s}_shap', self.feature_names))
        feature_value_names = list(map(lambda s: f'{s}_value', self.feature_names))
        df_shap = pd.DataFrame(self.explanation_values, columns=feature_shap_names)
        df_value = pd.DataFrame(self.feature_values, columns=feature_value_names)
        df = pd.concat([df_shap, df_value], axis=1)
        for i in range(len(self.feature_names)):
            feature_name = self.feature_names[i]
            geocp_result = self.geocp_results[i]
            df[f'{feature_name}_geo_uncertainty'] = geocp_result.geo_uncertainty
            df[f'{feature_name}_uncertainty'] = geocp_result.uncertainty
            df[f'{feature_name}_upper_bound'] = geocp_result.upper_bound
            df[f'{feature_name}_lower_bound'] = geocp_result.lower_bound
            df[f'{feature_name}_coverage_probability'] = geocp_result.coverage_probability
            df[f'{feature_name}_pred'] = geocp_result.pred
            df[f'{feature_name}_value'] = self.feature_values[:, i]
        df['x'] = self.coords[:, 0]
        df['y'] = self.coords[:, 1]
        return df

    def to_gdf(self) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(self.result, crs=self.crs,
                                               geometry=gpd.points_from_xy(x=self.result.x,
                                                                           y=self.result.y))
        return gdf

    def _shap_var(self) -> np.ndarray:
        return np.var(self.explanation_values, axis=0)

    def _predicted_shap_var(self) -> np.ndarray:
        predicted_shap_var_list = []
        for i in range(len(self.geocp_results)):
            geocp_results = self.geocp_results[i]
            predicted_shap_var = np.var(geocp_results.pred)
            predicted_shap_var_list.append(predicted_shap_var)
        return np.array(predicted_shap_var_list)


    def accuracy_summary(self) -> pd.DataFrame:
        coverage_proba_list = []
        for name in self.feature_names:
            coverage_name = f'{name}_coverage_probability'
            coverage_proba = self.result[coverage_name][0]
            coverage_proba_list.append(coverage_proba)
        coverage_proba_list = np.array(coverage_proba_list).reshape(-1, 1)
        shap_var = self._shap_var()
        pred_shap_var = self._predicted_shap_var()
        df = pd.DataFrame(np.hstack((coverage_proba_list,
                                     self.regression_scores.reshape(-1, 1),
                                     self.regression_rmse.reshape(-1, 1),
                                     shap_var.reshape(-1, 1),
                                     pred_shap_var.reshape(-1, 1))), columns=['coverage_probability', 'R2', 'RMSE', 'SHAP_Var', 'Pred_SHAP_Var'])
        df.index = self.feature_names
        return df

    def plot_absolute_shap_value_with_uncertainty(self, filename: str = None):
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
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_shap_values_with_uncertainty(self, i: int, filename: str = None):
        plt.rcParams['font.size'] = 12
        fig, ax = plt.subplots(figsize=(10, 10))
        result_i = self.result.iloc[i]
        feature_values_i = self.feature_values[i, :]
        shap_values_i = self.explanation_values[i, :]
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

        num_feature_names = len(self.feature_names)
        width = (num_feature_names / 10) * 0.06
        for y, x, low, high in zip(y_positions, shap_values_i, lower_bound_list, upper_bound_list):
            ax.plot([low, high], [y, y], color='#454545', linewidth=2, solid_capstyle='butt', zorder=1)
            ax.plot([low, low], [y - width, y + width], color='#454545', linewidth=2, solid_capstyle='butt', zorder=1)
            ax.plot([high, high], [y - width, y + width], color='#454545', linewidth=2, solid_capstyle='butt', zorder=1)
        plt.xlabel('Importance')

        y_ticks = []
        for name, value in zip(self.feature_names, feature_values_i):
            y_ticks.append(f'{name} = {value}')

        plt.yticks(y_positions, y_ticks)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_geo_uncertainty(self, max_cols: int = 5, figsize: List[int] = None, crs: Any = gcrs.WebMercator(), filename: str = None, shrink: float = 0.8):
        k = len(self.feature_names)
        n_cols = min(k, max_cols)
        n_rows = ceil(k / n_cols)

        if figsize is None:
            figsize = [30, n_rows * 5]

        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize, subplot_kw={'projection': crs})
        for i in range(len(self.feature_names)):
            row = int(i // n_cols)
            col = i - row * n_cols
            ax = axes[row][col]

            name = self.feature_names[i]

            ax.set_title(name)

            gplt.webmap(self.result_geo, projection=crs, provider=cx.providers.CartoDB.Voyager, ax=ax)

            ax.set_axis_on()

            gplt.pointplot(self.result_geo, hue=f'{name}_geo_uncertainty', cmap='Reds', legend=True,
                           legend_kwargs={'shrink': shrink}, ax=ax)
            plt.tight_layout()

        for ax in axes.flat:
            if not ax.has_data():  # Check if the subplot contains data
                fig.delaxes(ax)
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_partial_dependence_with_fitted_bounds(self, max_cols: int = 5, figsize: List[int] = None, n_splines: int = 50, filename: str = None):
        k = len(self.feature_names)
        n_cols = min(k, max_cols)
        n_rows = ceil(k / n_cols)

        if figsize is None:
            figsize = [30, n_rows * 5]

        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize)

        for i in range(len(self.feature_names)):
            row = int(i // n_cols)
            col = i - row * n_cols
            ax = axes[row][col]
            name = self.feature_names[i]
            shap_values = self.result[f'{name}_shap'].values
            feature_values = self.result[f'{name}_value'].values
            lower_bounds = self.result[f'{name}_lower_bound'].values
            upper_bounds = self.result[f'{name}_upper_bound'].values
            lam = np.logspace(2, 7, 5).reshape(-1, 1)
            upper_gam = LinearGAM(n_splines=n_splines, fit_intercept=False).gridsearch(feature_values.reshape(-1, 1),
                                                                              upper_bounds.reshape(-1, 1), lam=lam)
            lower_gam = LinearGAM(n_splines=n_splines, fit_intercept=False).gridsearch(feature_values.reshape(-1, 1),
                                                                              lower_bounds.reshape(-1, 1), lam=lam)
            x = np.linspace(feature_values.min(), feature_values.max(), 250)
            y_pred_lower = upper_gam.predict(x)
            y_pred_upper = lower_gam.predict(x)
            ax.fill_between(x, y_pred_upper, y_pred_lower, color='#3594cc', alpha=0.5)
            ax.scatter(feature_values, shap_values, s=5, c='#d8a6a6')
            # ax.scatter(feature_values, upper_bounds, s=5, c='#3594cc')
            ax.set_ylabel(f'Shapley Value - {name}')
            ax.set_xlabel(f'Feature Value - {name}')
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_partial_plot_with_individual_intervals(self, max_cols: int = 5, figsize: List[int] = None, filename: str = None):
        k = len(self.feature_names)
        n_cols = min(k, max_cols)
        n_rows = ceil(k / n_cols)

        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize)

        for i in range(len(self.feature_names)):
            row = int(i // n_cols)
            col = i - row * n_cols
            ax = axes[row][col]
            name = self.feature_names[i]
            shap_values = self.result[f'{name}_shap'].values
            feature_values = self.result[f'{name}_value'].values
            lower_bounds = self.result[f'{name}_lower_bound'].values
            upper_bounds = self.result[f'{name}_upper_bound'].values
            pred_values = self.result[f'{name}_pred'].values
            ax.scatter(feature_values, pred_values, s=2, c='#8cc5e3', zorder=1)
            for x, low, high in zip(feature_values, lower_bounds, upper_bounds):
                ax.plot([x, x], [low, high], color='#3594cc', linewidth=0.8, solid_capstyle='butt', zorder=1)
            ax.scatter(feature_values, shap_values, s=2, c='#d8a6a6', zorder=10)
            ax.set_ylabel(f'Shapley Value - {name}')
            ax.set_xlabel(f'Feature Value - {name}')
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()





