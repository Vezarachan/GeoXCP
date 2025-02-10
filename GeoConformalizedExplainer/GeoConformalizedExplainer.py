from typing import Callable, Any, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import math
import shap
import torch
from examples.plot_qrf_prediction_intervals import train_index
from fastshap import KernelExplainer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
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
from GeoConformalizedExplainer.model import get_dataloader, MultipleTargetRegression
import torch.nn as nn
import torch.optim as optim


class GeoConformalizedExplainerResults:
    def __init__(self, explanation: np.ndarray,
                 geocp_results: List[GeoConformalResults],
                 regression_r2: np.ndarray,
                 regression_rmse: np.ndarray,
                 coords: np.ndarray,
                 feature_values: np.ndarray,
                 feature_names: List[str],
                 crs: str = 'EPSG:4326'):
        self.explanation_values = explanation
        self.feature_names = feature_names
        self.geocp_results = geocp_results
        self.coords = coords
        self.regression_r2 = regression_r2
        self.regression_rmse = regression_rmse
        self.crs = crs
        self.feature_values = feature_values
        self.result = self._get_shap_values_with_uncertainty()
        self.result_geo = self.to_gdf()
        self.K = len(self.feature_names)

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
            df[f'{feature_name}_shap_abs'] = np.abs(df[f'{feature_name}_shap'])
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

    def get_svc(self, cols: List, coef_type: str = 'gwr', bw_min: int = 5, bw_max: int = 50, include_geo_effects: bool = True):
        N, _ = self.feature_values.shape
        params = np.zeros((N, self.K))
        for k in range(self.K):
            params[:, k] = self.geocp_results[k].pred

        for i in cols:
            if coef_type == 'raw':
                params[:, i] = params[:, i] / (self.feature_values - self.feature_values.mean(axis=0))[:, i]
            if coef_type == 'gwr':
                try:
                    import mgwr
                except ImportError:
                    print("Please install mgwr package (e.g., pip install mgwr).")
                y = params[:, i].reshape(-1, 1)
                X =(self.feature_values - self.feature_values.mean(axis=0))[:, i].reshape(-1, 1)
                gwr_selector = mgwr.sel_bw.Sel_BW(self.coords, y, X)
                gwr_bw = gwr_selector.search(bw_min=bw_min, bw_max=bw_max)
                gwr_model = mgwr.gwr.GWR(self.coords, y, X, gwr_bw).fit()
                params[:, i] = gwr_model.params[:, 1]
        return params[:, cols]


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
                                     self.regression_r2.reshape(-1, 1),
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

    def _format_number_based_on_magnitude(self, num) -> str:
        if num == 0:
            return "0"
        magnitude = math.floor(math.log10(abs(num)))  # Compute order of magnitude
        decimal_places = max(0, 2 - magnitude)  # Determine the number of decimal places based on the order of magnitude
        formatted_number = f"{num:.{decimal_places}f}"  # Dynamically format the number
        return formatted_number

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
        # labels = [f'+{self._format_number_based_on_magnitude(e)}' if e >= 0 else f'{self._format_number_based_on_magnitude(e)}' for e in shap_values_i]

        bars = ax.barh(self.feature_names, shap_values_i, color=colors)
        y_positions = np.arange(len(self.feature_names))

        for i, (bar, value) in enumerate(zip(bars, shap_values_i)):
            offset = -10 if value < 0 else 1
            label = f'+{self._format_number_based_on_magnitude(value)}' if value >= 0 else f'{self._format_number_based_on_magnitude(value)}'
            color = '#ff0d57' if value >= 0 else '#1e88e5'
            ax.annotate(
                label,
                xy=(value, i + bar.get_height() / 3),  # Position
                xytext=(offset * len(label), 0),  # Offset (x, y) in points
                textcoords="offset points",  # Relative positioning
                va='center',  # Vertical alignment
                fontsize=8,
                color=color
            )

        num_feature_names = len(self.feature_names)
        width = (num_feature_names / 10) * 0.05
        for y, x, low, high in zip(y_positions, shap_values_i, lower_bound_list, upper_bound_list):
            mid = (low + high) / 2
            ax.plot([low, high], [y, y], color='#454545', linewidth=1.5, solid_capstyle='butt', zorder=1)
            ax.plot([low, low], [y - width, y + width], color='#454545', linewidth=1.5, solid_capstyle='butt', zorder=1)
            ax.plot([high, high], [y - width, y + width], color='#454545', linewidth=1.5, solid_capstyle='butt', zorder=1)
            ax.plot(mid, y, color='#ea801c', marker='o', markersize=6, zorder=2)
        plt.xlabel('Importance')

        y_ticks = []
        for name, value in zip(self.feature_names, feature_values_i):
            y_ticks.append(f'{name} = {value}')

        plt.yticks(y_positions, y_ticks)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_geo_uncertainty(self, max_cols: int = 5, figsize: List[int] = None, crs: Any = gcrs.WebMercator(), filename: str = None, shrink: float = 0.8, basemap: bool = True, s_limits: List[int] = (2, 12), cmap: str = 'flare'):
        n_cols = min(self.K, max_cols)
        n_rows = ceil(self.K / n_cols)

        if figsize is None:
            figsize = [30, n_rows * 5]

        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize, subplot_kw={'projection': crs})
        for i in range(len(self.feature_names)):
            row = int(i // n_cols)
            col = i - row * n_cols
            ax = axes[row][col]

            name = self.feature_names[i]

            ax.set_title(name)

            if basemap:
                gplt.webmap(self.result_geo, projection=crs, provider=cx.providers.CartoDB.Voyager, ax=ax)

            ax.set_axis_on()

            gplt.pointplot(self.result_geo, legend_var='hue', hue=f'{name}_geo_uncertainty', scale=f'{name}_shap_abs', cmap=cmap, limits=s_limits, legend=True,
                           legend_kwargs={'shrink': shrink}, ax=ax)
            plt.tight_layout()

        for ax in axes.flat:
            if not ax.has_data():  # Check if the subplot contains data
                fig.delaxes(ax)
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()



    def plot_partial_dependence_with_fitted_bounds(self, max_cols: int = 5, figsize: List[int] = None, n_splines: int = 50, filename: str = None):
        n_cols = min(self.K, max_cols)
        n_rows = ceil(self.K / n_cols)

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
        for ax in axes.flat:
            if not ax.has_data():  # Check if the subplot contains data
                fig.delaxes(ax)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_partial_plot_with_individual_intervals(self, max_cols: int = 5, figsize: List[int] = None, filename: str = None):
        n_cols = min(self.K, max_cols)
        n_rows = ceil(self.K / n_cols)

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
            pred_values = self.result[f'{name}_pred'].values
            ax.scatter(feature_values, pred_values, s=2, c='#8cc5e3', zorder=1)
            for x, low, high in zip(feature_values, lower_bounds, upper_bounds):
                ax.plot([x, x], [low, high], color='#3594cc', linewidth=0.8, solid_capstyle='butt', zorder=1)
            ax.scatter(feature_values, shap_values, s=2, c='#d8a6a6', zorder=10)
            ax.set_ylabel(f'Shapley Value - {name}')
            ax.set_xlabel(f'Feature Value - {name}')

        for ax in axes.flat:
            if not ax.has_data():  # Check if the subplot contains data
                fig.delaxes(ax)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

class GeoConformalizedExplainer:
    """
    Spatial Explanation under Uncertainty
    Geographically Conformalized Explanations for Black-Box Models
    """
    def __init__(self,
                 prediction_f: Callable,
                 x_train: Union[np.ndarray, pd.DataFrame],
                 x_calib: Union[np.ndarray, pd.DataFrame],
                 coord_calib: Union[np.ndarray, pd.DataFrame],
                 shap_value_f: Callable = None,
                 miscoverage_level: float = 0.1,
                 band_width: Union[List[float], np.ndarray, float] = None,
                 feature_names: Union[List[str], np.ndarray] = None,
                 is_single_model: bool = True):
        """
        :param prediction_f:
        :param shap_value_f:
        :param x_train:
        :param x_calib:
        :param coord_calib:
        :param miscoverage_level:
        :param band_width:
        :param feature_names:
        :param is_single_model:
        """
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.prediction_f = prediction_f
        self.shap_value_f = shap_value_f
        _, k = x_calib.shape
        self.num_variables = k
        self.coord_calib = coord_calib
        self.miscoverage_level = miscoverage_level
        self.band_width = band_width
        self.is_single_model = is_single_model
        if feature_names is None:
            self.feature_names = [f'X{i}' for i in range(k)]
        else:
            self.feature_names = feature_names
        if isinstance(x_train, pd.DataFrame):
            self.x_train = x_train.values
        else:
            self.x_train = x_train
        if isinstance(x_calib, pd.DataFrame):
            self.x_calib = x_calib.values
        else:
            self.x_calib = x_calib
    def _compute_explanation_values(self, x: np.ndarray) -> np.ndarray:
        """
        Compute explanation values with explanation methods such as SHAP. LIME, etc.
        :param x:
        :return:
        """
        # KernelSHAP
        # background = shap.sample(x, 200)
        # explainer = shap.KernelExplainer(self.prediction_f, background, feature_names=self.feature_names)
        # explanation_result = explainer(x).values
        # ExactSHAP
        # explainer = shap.Explainer(self.prediction_f, self.x_train, feature_names=self.feature_names, algorithm='auto')
        # explanation_result = explainer(x).values
        if self.shap_value_f is None:
            # batch_size = x.shape[0] // 2
            # explainer = KernelExplainer(self.prediction_f, x)
            # explainer.stratify_background_set(20)
            # explanation_result = explainer.calculate_shap_values(x, verbose=True, outer_batch_size=batch_size, inner_batch_size=batch_size, background_fold_to_use=0)
            # explanation_result = explanation_result[:, :self.num_variables]
            background = shap.sample(x, 100)
            explainer = shap.KernelExplainer(self.prediction_f, background, feature_names=self.feature_names)
            explanation_result = explainer(x).values
        else:
            explanation_result = self.shap_value_f(x)
        return explanation_result

    def _fit_explanation_value_predictor(self, x: np.ndarray, t: np.ndarray, s: np.ndarray) -> XGBRegressor:
        """
        Fit the regression model between explanation values of a variable and input values X, predicted values t.
        Each feature has its own regression model.
        :param X:
        :param t: prediction values by the black-box model
        :param s: ground truth explanation values
        :return:
        """
        params = {'eta': [0.01, 0.1, 1.0], 'gamma': [0, 0.1], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0],
                  'n_estimators': [100, 200, 500], 'max_depth': [2, 4],
                  'min_child_weight': [1, 2], 'nthread': [2]}
        kf = KFold(n_splits=3, random_state=1, shuffle=True)
        model = XGBRegressor()
        model_cv = GridSearchCV(estimator=model,
                                 param_grid=params,
                                 verbose=0,
                                 return_train_score=False,
                                 n_jobs=8,
                                 cv=kf
                                )
        t = t.reshape(-1, 1)
        x_new = np.hstack((x, t))
        model_cv.fit(x_new, s)
        regressor = XGBRegressor(params=model_cv.best_params_)
        regressor.fit(x_new, s)
        return regressor

    def _fit_explanation_value_predictor_single_model(self, x: np.ndarray, t: np.ndarray, s: np.ndarray) -> MLPRegressor:
        """
        Fit the regression model between explanation values of a variable and input values X, predicted values t.
        All features are fitted with a single regression model.
        :param x:
        :param t:
        :param s:
        :return:
        """
        model = MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024, 1024, 1024),
                             max_iter=2000,
                             activation='relu',
                             solver='adam',
                             verbose=False,
                             learning_rate='adaptive',
                             early_stopping=True,
                             alpha=0.01)
        t = t.reshape(-1, 1)
        x_new = np.hstack((x, t))
        x_new = self.scaler.fit_transform(x_new)
        x_new = self.imputer.fit_transform(x_new)
        model.fit(x_new, s)
        return model

    def _fit_explanation_value_predictor_nn_model(self, x_train: np.ndarray, t_train: np.ndarray, s_train: np.ndarray, x_val: np.ndarray, t_val: np.ndarray, s_val: np.ndarray, is_save: bool = False) -> MultipleTargetRegression:
        t_train = t_train.reshape(-1, 1)
        x_train_new = np.hstack((x_train, t_train))
        x_train_new = self.scaler.fit_transform(x_train_new)
        x_train_new = self.imputer.fit_transform(x_train_new)
        t_val = t_val.reshape(-1, 1)
        x_val_new = np.hstack((x_val, t_val))
        x_val_new = self.scaler.fit_transform(x_val_new)
        x_val_new = self.imputer.fit_transform(x_val_new)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader = get_dataloader(x_train_new, s_train, 100, num_workers=2)
        val_dataloader = get_dataloader(x_val_new, s_val, 100, num_workers=2)
        model = MultipleTargetRegression(input_dim=x_train_new.shape[1], output_dim=s_train.shape[1], num_blocks=2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 500
        train_loss = 0.0

        for epoch in tqdm(range(num_epochs)):
            model.train()
            for batch_X, batch_y in train_dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():  # No gradients needed during evaluation
                for batch_X, batch_y in val_dataloader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_dataloader):.4f}, Val Loss: {val_loss/len(val_dataloader):.4f}")
        if is_save:
            torch.save(model.state_dict(), f'../data/model_trained.pth')
        return model

    def _predict_explanation_values(self, x: np.ndarray, model: MultipleTargetRegression) -> np.ndarray:
        """
        Predict explanation values from input values X and predicted values t.
        :param X:
        :param t:
        :return:
        """
        model.eval()
        x = self.scaler.fit_transform(x)
        x = torch.tensor(x, dtype=torch.float32)
        y_pred = model(x)
        y_pred = y_pred.detach().cpu().numpy()
        return y_pred

    def _explain_ith_variable(self, i: int, s_train: np.ndarray, t_train: np.ndarray, x_test_new: np.ndarray, s_test: np.ndarray, x_calib_new: np.ndarray, s_calib: np.ndarray, coord_test: np.ndarray) -> Tuple[Any, float, float]:
        """
        Explain a single variable. Each variable has its own regression model.
        :param i:
        :param s_train:
        :param t_train:
        :param x_test_new:
        :param s_test:
        :param x_calib_new:
        :param s_calib:
        :return:
        """
        regressor = self._fit_explanation_value_predictor(self.x_train, t_train, s_train[:, i])
        r2 = regressor.score(x_test_new, s_test[:, i])
        rmse = root_mean_squared_error(regressor.predict(x_test_new), s_test[:, i])
        geocp = GeoConformalSpatialPrediction(predict_f=regressor.predict,
                                              miscoverage_level=self.miscoverage_level,
                                              bandwidth=self.band_width,
                                              coord_calib=self.coord_calib,
                                              coord_test=coord_test,
                                              X_calib=x_calib_new, y_calib=s_calib[:, i],
                                              X_test=x_test_new, y_test=s_test[:, i])
        result_ith_variable = geocp.analyze()
        return result_ith_variable, r2, rmse

    def _explain_variables_in_single_model(self, s_train: np.ndarray, t_train: np.ndarray, x_test_new: np.ndarray, s_test: np.ndarray, x_calib_new: np.ndarray, s_calib: np.ndarray, coord_test: np.ndarray) -> List[List[Any]]:
        """
        Explain all variables in a single model.
        :param s_train:
        :param t_train:
        :param x_test_new:
        :param s_test:
        :param x_calib_new:
        :param s_calib:
        :return:
        """
        regressor = self._fit_explanation_value_predictor_single_model(self.x_train, t_train, s_train)
        R2s = r2_score(s_test, regressor.predict(self.scaler.fit_transform(x_test_new)), multioutput='raw_values')
        RMSEs = root_mean_squared_error(regressor.predict(self.scaler.fit_transform(x_test_new)), s_test, multioutput='raw_values')
        results = []
        for i in range(self.num_variables):
            geocp = GeoConformalSpatialPrediction(predict_f=lambda x_: regressor.predict(self.scaler.fit_transform(x_))[:, i],
                                                  miscoverage_level=self.miscoverage_level,
                                                  bandwidth=self.band_width,
                                                  coord_calib=self.coord_calib,
                                                  coord_test=coord_test,
                                                  X_calib=x_calib_new, y_calib=s_calib[:, i],
                                                  X_test=x_test_new, y_test=s_test[:, i])
            result_ith_variable = geocp.analyze()
            results.append([result_ith_variable, R2s[i], RMSEs[i]])
        return results

    def _explain_variables_in_nn_model(self, s_train: np.ndarray, t_train: np.ndarray, x_test_new: np.ndarray, s_test: np.ndarray, x_calib_new: np.ndarray, s_calib: np.ndarray, coord_test: np.ndarray) -> List[List[Any]]:
        regressor = self._fit_explanation_value_predictor_nn_model(self.x_train, t_train, s_train)
        s_test_pred = self._predict_explanation_values(x_test_new, regressor)
        R2s = r2_score(s_test, s_test_pred, multioutput='raw_values')
        RMSEs = root_mean_squared_error(s_test, s_test_pred, multioutput='raw_values')
        results = []
        for i in range(self.num_variables):
            geocp = GeoConformalSpatialPrediction(predict_f=lambda x_: self._predict_explanation_values(x_, regressor)[:, i],
                                                  miscoverage_level=self.miscoverage_level,
                                                  bandwidth=self.band_width,
                                                  coord_calib=self.coord_calib,
                                                  coord_test=coord_test,
                                                  X_calib=x_calib_new, y_calib=s_calib[:, i],
                                                  X_test=x_test_new, y_test=s_test[:, i])
            result_ith_variable = geocp.analyze()
            results.append([result_ith_variable, R2s[i], RMSEs[i]])
        return results


    def uncertainty_aware_explain(self, x_test: Union[np.ndarray, pd.DataFrame], coord_test: Union[np.ndarray, pd.DataFrame], n_jobs: int = 4, is_geo: bool = False) -> GeoConformalizedExplainerResults:
        """
        Explain black-box model with uncertainty aware method.
        :param x_test:
        :param coord_test:
        :param n_jobs:
        :param is_geo:
        :return:
        """
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values
        if isinstance(coord_test, pd.DataFrame):
            coord_test = coord_test.values
        t_train = self.prediction_f(self.x_train)
        print('Training SHAP')
        s_train = self._compute_explanation_values(self.x_train)
        t_calib = self.prediction_f(self.x_calib).reshape(-1, 1)
        print('Calibrating SHAP')
        s_calib = self._compute_explanation_values(self.x_calib)
        x_calib_new = np.hstack((self.x_calib, t_calib))
        print('Testing SHAP')
        s_test = self._compute_explanation_values(x_test)
        t_test = self.prediction_f(x_test).reshape(-1, 1)
        x_test_new = np.hstack((x_test, t_test))
        print('Explaining Variables')
        if self.is_single_model:
            results = self._explain_variables_in_nn_model(s_train, t_train, x_test_new, s_test, x_calib_new, s_calib, coord_test)
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._explain_ith_variable)(i, s_train, t_train, x_test_new, s_test, x_calib_new, s_calib, coord_test) for i
                in tqdm(range(self.num_variables)))
        geocp_results = [result[0] for result in results]
        r2 = np.array([result[1] for result in results])
        rmse = np.array([result[2] for result in results])
        return GeoConformalizedExplainerResults(explanation=s_test, geocp_results=geocp_results, regression_r2=r2, regression_rmse=rmse, coords=coord_test, feature_values=x_test, feature_names=self.feature_names)








