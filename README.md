# Car Price Prediction

An interactive Streamlit application that estimates car prices from vehicle details using saved regression model artifacts. It presents each model-file prediction, their arithmetic average, and a side-by-side comparison.

## Features

- Estimate a car price in USD from its details.
- Show prediction outputs from four saved model files.
- Display an arithmetic average of the four outputs.
- Compare model outputs in a table and chart.
- Enter vehicle details through an interactive sidebar form.
- Use a professional automotive-themed Streamlit interface.
- Run locally with a small set of Python dependencies.

## Machine Learning Models

The application has four model slots:

- **Linear Regression**: a linear regression model is intended for this slot.
- **Decision Tree Regressor**: predicts using decision rules learned from feature values.
- **KNN Regressor**: predicts from nearby examples in the feature space.
- **Random Forest Regressor**: combines predictions from an ensemble of decision trees.

**Artifact note:** The file `linear_regression.pkl` currently deserializes as a `RandomForestRegressor`, not a `LinearRegression`. The app warns about this and still uses the artifact under its supplied file label. As a result, the current files do not provide four distinct estimator types; replace this artifact with a correctly trained linear regression model if that is intended. No model is described as best because the repository does not include supporting evaluation results.

## Input Features

The form uses these ten predictor features from the dataset:

- `Brand`
- `Model_Year`
- `Kilometers_Driven`
- `Fuel_Type`
- `Transmission`
- `Owner_Type`
- `Engine_CC`
- `Max_Power_bhp`
- `Mileage_kmpl`
- `Seats`

The models expect numeric features plus one-hot encoded categories. The app uses `pandas.get_dummies` with `drop_first=True`, then aligns the resulting columns to the saved models' `feature_names_in_` schema. `Car_ID` is present in the CSV but is not used as a predictor.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Project Structure

```text
CAR_PRICE_PREDICTION/
|-- .gitignore
|-- README.md
|-- app.py
|-- assets/
|   `-- car_banner.jpg
|-- car_price_dataset_medium.csv
|-- Decision_tree.pkl
|-- KNN.pkl
|-- linear_regression.pkl
|-- Random_forest.pkl
`-- requirements.txt
```

## Installation

Use Python 3.10 or newer. Replace `<repository-url>` with the URL of your GitHub repository.

```bash
git clone <repository-url> Car_Price_Prediction
cd Car_Price_Prediction
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app expects the CSV, four model files, and `assets/car_banner.jpg` to remain in the project structure shown above. Model files are loaded using paths relative to `app.py`.

## How It Works

1. The user enters car details in the sidebar.
2. The app builds a one-row input and applies the categorical encoding expected by the saved estimators.
3. The input columns are aligned to each model's saved feature schema.
4. All four model files generate an individual price prediction.
5. The app calculates the arithmetic mean of those four outputs.
6. Individual outputs are shown as metrics and in a comparison table and chart.

## Dataset

The repository includes `car_price_dataset_medium.csv`, a 1,000-row, 12-column CSV. Its target column is `Price_USD`. It also contains `Car_ID`, the ten input features listed above, and the target; `Car_ID` is excluded from model inputs. The app uses the CSV to populate categorical options and set initial numeric values.

## Model Training

The repository contains serialized model artifacts for inference, loaded with Joblib. The feature names stored in the artifacts match the app's one-hot encoded input schema. Training scripts and evaluation results are not included, so the exact training run, data split, and model metrics cannot be reproduced from this repository alone.

## Future Improvements

- Add reproducible training and evaluation scripts.
- Tune models and report cross-validated metrics.
- Expand the training data and investigate additional feature engineering.
- Add a model evaluation dashboard.
- Deploy the app to a cloud Streamlit host.
- Integrate current market data where a reliable data source is available.

## Disclaimer

Predicted prices are estimates based on the saved model artifacts. They are not guaranteed market prices and should not replace an independent vehicle inspection or valuation.

## Author

**Dhruvil**  
Artificial Intelligence & Data Science Engineering