# GeoXCP

> **Geo**spatial e**X**planation **C**onformal **P**rediction (GeoXCP) is a powerful framework to measure uncertainty of spatial explanations.


## Usage
```python
from GeoConformalizedExplainer import GeoConformalizedExplainer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('...')
X_geo = data[['X1', 'X2', 'X3', 'X4', 'X5']]
y = data['y']
loc = data[['lon', 'lat']]

X_train, X_temp, y_train, y_temp, _, loc_temp = train_test_split(X_geo, y, loc, train_size=0.8, random_state=42)
X_calib, X_test, y_calib, y_test, loc_calib, loc_test = train_test_split(X_geo, y, loc_temp, train_size=0.5, random_state=42)
feature_names = X_geo.columns

# Fit a XGBoost Regressor model
model = XGBRegressor().fit(X_train, y_train)

# Initialize a GeoXCP Explainer
explainer = GeoConformalizedExplainer(prediction_f=model.predict,
                                      x_train=X_train,
                                      x_calib=X_calib,
                                      coord_calib=loc_calib.values,
                                      miscoverage_level=0.1,
                                      band_width=0.3,
                                      feature_names=feature_names)

# Explain the data with uncertainty
results = explainer.uncertainty_aware_explain(x_test=X_test, coord_test=loc_test)

# Make an accuracy summary
results.accuracy_summary()

# Convert the results into GeoPandas GeoDataFrame format
pred_results = results.result_geo

# Plot the shap values with uncertainty for certain location
results.plot_shap_values_with_uncertainty(i=20)

# Plot the spatial distribution of uncertainty
results.plot_geo_uncertainty()
```
