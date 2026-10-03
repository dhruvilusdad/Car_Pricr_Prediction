from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_CANDIDATES = [
	PROJECT_DIR / "car_price_dataset_medium.csv",
	PROJECT_DIR / "car_price_dataset.csv",
]
MODEL_FILES = {
	"Linear Regression": "linear_regression.pkl",
	"Decision Tree": "Decision_tree.pkl",
	"KNN": "KNN.pkl",
	"Random Forest": "Random_forest.pkl",
}
CATEGORICAL_FEATURES = ["Brand", "Fuel_Type", "Transmission", "Owner_Type"]
NUMERIC_FEATURES = [
	"Model_Year",
	"Kilometers_Driven",
	"Engine_CC",
	"Max_Power_bhp",
	"Mileage_kmpl",
	"Seats",
]
REQUIRED_COLUMNS = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["Price_USD"])

st.set_page_config(
	page_title="Car Price Prediction",
	page_icon="🚗",
	layout="wide",
)

# Keep the visual treatment restrained so model outputs stay the focus.
st.markdown(
	"""
	<style>
	.stApp {
		background: #f4f7f8;
		color: #1b2f30 !important;
	}
	.stApp p,
	.stApp label,
	.stApp h1,
	.stApp h2,
	.stApp h3,
	.stApp h4,
	.stApp [data-testid="stMarkdownContainer"],
	.stApp [data-testid="stWidgetLabel"],
	.stApp [data-testid="stWidgetLabel"] *,
	.stApp [data-testid="stMetricLabel"],
	.stApp [data-testid="stMetricValue"],
	.stApp [data-testid="stMetricDelta"],
	.stApp [data-testid="stCaptionContainer"],
	.stApp [data-testid="stDataFrame"] *,
	.stApp [data-testid="stTable"] *,
	[role="listbox"],
	[role="listbox"] *,
	[role="option"],
	[role="option"] *,
	[data-baseweb="popover"],
	[data-baseweb="popover"] * {
		color: #1b2f30 !important;
	}
	.stApp input,
	.stApp textarea,
	.stApp [data-baseweb="input"] input,
	.stApp [data-baseweb="select"] * {
		color: #ffffff !important;
	}
	.stApp [data-baseweb="select"] svg {
		color: #ffffff !important;
		fill: #ffffff !important;
	}
	.stApp input[type="number"]::-webkit-inner-spin-button,
	.stApp input[type="number"]::-webkit-outer-spin-button {
		color: #ffffff !important;
		opacity: 1;
	}
	[role="listbox"],
	[role="listbox"] *,
	[role="option"],
	[role="option"] *,
	[data-testid="stSelectboxVirtualDropdown"] * {
		color: #ffffff !important;
	}
	.stApp [data-testid="stFormSubmitButton"] button {
		color: #ffffff !important;
	}
	[data-testid="stHeader"] { background: rgba(244, 247, 248, 0.92); }
	[data-testid="stSidebar"] { background: #eef2f2; }
	.block-container { padding-top: 2.2rem; padding-bottom: 1rem; }
	[data-testid="stMetric"] {
		background: #ffffff;
		border: 1px solid #dce5e5;
		border-top: 3px solid #0b7775;
		padding: 1rem 1.1rem;
		border-radius: 6px;
	}
	[data-testid="stMetricValue"] { color: #123b40 !important; }
	.app-footer {
		color: #405356 !important;
		border-top: 1px solid #dce5e5;
		margin-top: 2.5rem;
		padding-top: 1rem;
		font-size: 0.85rem;
		text-align: center;
	}
	.stApp [data-testid="stImage"] {
		width: 100%;
		max-width: 920px;
		margin: 1.4rem auto 0;
	}
	.stApp [data-testid="stImage"] img {
		width: 100%;
		height: auto;
		border-radius: 10px;
		box-shadow: 0 12px 28px rgba(13, 36, 39, 0.18);
	}
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
	"""Load the local training dataset used to recover category levels."""
	dataset_path = next((path for path in DATASET_CANDIDATES if path.is_file()), None)
	if dataset_path is None:
		raise FileNotFoundError(
			"Dataset not found. Add car_price_dataset.csv to the project folder."
		)
	dataset = pd.read_csv(dataset_path)
	missing_columns = REQUIRED_COLUMNS.difference(dataset.columns)
	if missing_columns:
		raise ValueError(
			"Dataset is missing required columns: " + ", ".join(sorted(missing_columns))
		)
	for column in CATEGORICAL_FEATURES:
		dataset[column] = dataset[column].astype("string").str.strip()
	return dataset


@st.cache_resource
def load_models() -> tuple[dict[str, object], list[str]]:
	"""Load each saved estimator and verify that their expected schemas agree."""
	models = {}
	expected_features = None
	for display_name, filename in MODEL_FILES.items():
		model_path = PROJECT_DIR / filename
		if not model_path.is_file():
			raise FileNotFoundError(f"Model file not found: {filename}")
		try:
			model = joblib.load(model_path)
		except Exception as error:
			raise RuntimeError(f"Could not load {filename}: {error}") from error

		feature_names = getattr(model, "feature_names_in_", None)
		if feature_names is None:
			raise ValueError(
				f"{filename} does not include feature_names_in_; its input schema "
				"cannot be safely reconstructed."
			)
		feature_names = list(feature_names)
		if expected_features is not None and feature_names != expected_features:
			raise ValueError("The saved models do not share the same input feature schema.")
		expected_features = feature_names
		models[display_name] = model
	return models, expected_features or []


st.title("🚗 Car Price Prediction")
st.write("Predict car prices using multiple Machine Learning models.")

try:
	dataset = load_dataset()
	models, model_features = load_models()
except Exception as error:
	st.error(f"The application could not prepare its data or models. {error}")
	st.stop()

# This artifact's filename says Linear Regression, but its serialized class is a forest.
linear_model_type = type(models["Linear Regression"]).__name__
if linear_model_type != "LinearRegression":
	st.warning(
		"The file `linear_regression.pkl` contains a "
		f"{linear_model_type}, not a LinearRegression estimator. Its prediction is "
		"shown under the supplied file label; replace the artifact if that is not intended."
	)

# Category levels and numerical defaults come from the colocated dataset.
category_options = {
	column: sorted(dataset[column].dropna().astype(str).unique().tolist())
	for column in CATEGORICAL_FEATURES
}
if any(not options for options in category_options.values()):
	st.error("The dataset must contain at least one value for every categorical feature.")
	st.stop()

with st.sidebar:
	st.header("Car Details")
	with st.form("car_details_form"):
		brand = st.selectbox("Brand", category_options["Brand"])
		current_year = date.today().year
		model_year = st.number_input(
			"Model Year",
			min_value=1980,
			max_value=current_year + 1,
			value=min(max(int(dataset["Model_Year"].median()), 1980), current_year + 1),
			step=1,
		)
		kilometers = st.number_input(
			"Kilometers Driven",
			min_value=0,
			max_value=1_000_000,
			value=max(0, int(dataset["Kilometers_Driven"].median())),
			step=1_000,
		)
		fuel_type = st.selectbox("Fuel Type", category_options["Fuel_Type"])
		transmission = st.selectbox("Transmission", category_options["Transmission"])
		owner_type = st.selectbox("Owner Type", category_options["Owner_Type"])
		engine_cc = st.number_input(
			"Engine CC",
			min_value=0.0,
			max_value=10_000.0,
			value=float(dataset["Engine_CC"].median()),
			step=100.0,
		)
		max_power = st.number_input(
			"Max Power (bhp)",
			min_value=0.0,
			max_value=2_000.0,
			value=float(dataset["Max_Power_bhp"].median()),
			step=10.0,
		)
		mileage = st.number_input(
			"Mileage (km/l)",
			min_value=0.0,
			max_value=200.0,
			value=float(dataset["Mileage_kmpl"].median()),
			step=1.0,
		)
		seats = st.number_input(
			"Seats",
			min_value=1,
			max_value=20,
			value=min(max(int(dataset["Seats"].median()), 1), 20),
			step=1,
		)
		predict_clicked = st.form_submit_button(
			"Predict Price", type="primary", width="stretch"
		)

if predict_clicked:
	# Validate explicitly as well as constraining the widgets, so invalid values fail clearly.
	numeric_inputs = {
		"Model Year": model_year,
		"Kilometers Driven": kilometers,
		"Engine CC": engine_cc,
		"Max Power (bhp)": max_power,
		"Mileage (km/l)": mileage,
		"Seats": seats,
	}
	validation_errors = []
	if not 1980 <= model_year <= current_year + 1:
		validation_errors.append("Model Year must be between 1980 and next year.")
	for label in ["Kilometers Driven", "Engine CC", "Max Power (bhp)", "Mileage (km/l)"]:
		if numeric_inputs[label] < 0:
			validation_errors.append(f"{label} cannot be negative.")
	if not 1 <= seats <= 20:
		validation_errors.append("Seats must be between 1 and 20.")
	if not all(np.isfinite(float(value)) for value in numeric_inputs.values()):
		validation_errors.append("Enter valid finite numbers for all numeric fields.")

	if validation_errors:
		for validation_error in validation_errors:
			st.error(validation_error)
	else:
		input_row = pd.DataFrame(
			[
				{
					"Brand": brand,
					"Model_Year": model_year,
					"Kilometers_Driven": kilometers,
					"Fuel_Type": fuel_type,
					"Transmission": transmission,
					"Owner_Type": owner_type,
					"Engine_CC": engine_cc,
					"Max_Power_bhp": max_power,
					"Mileage_kmpl": mileage,
					"Seats": seats,
				}
			]
		)
		try:
			# Match pandas one-hot encoding used in training, then enforce model column order.
			for column in CATEGORICAL_FEATURES:
				input_row[column] = pd.Categorical(
					input_row[column], categories=category_options[column]
				)
			encoded_input = pd.get_dummies(
				input_row,
				columns=CATEGORICAL_FEATURES,
				drop_first=True,
				dtype=int,
			).reindex(columns=model_features, fill_value=0)
			if encoded_input.isna().any().any():
				raise ValueError("The supplied details produced missing model features.")

			predictions = {
				name: float(model.predict(encoded_input)[0])
				for name, model in models.items()
			}
			if not all(np.isfinite(value) for value in predictions.values()):
				raise ValueError("A model returned a non-finite price.")
		except Exception as error:
			st.error(f"Prediction failed. Check the saved models and their feature schema. {error}")
		else:
			st.subheader("Predicted Prices")
			metric_columns = st.columns(4)
			for column, (name, value) in zip(metric_columns, predictions.items()):
				column.metric(name, f"${value:,.2f}")

			average_price = float(np.mean(list(predictions.values())))
			st.subheader("Average Predicted Price")
			st.metric("Ensemble average", f"${average_price:,.2f}")

			st.subheader("Model Comparison")
			comparison = pd.DataFrame(
				{
					"Model": list(predictions.keys()),
					"Predicted Price (USD)": list(predictions.values()),
				}
			)
			st.dataframe(
				comparison.style.format({"Predicted Price (USD)": "${:,.2f}"}),
				hide_index=True,
				width="stretch",
			)
			st.bar_chart(
				comparison.set_index("Model"),
				y="Predicted Price (USD)",
				color="#0b7775",
			)
			st.image(PROJECT_DIR / "assets" / "car_banner.jpg", width="stretch")
else:
	st.info("Enter the vehicle details in the sidebar and select Predict Price to compare models.")

st.markdown(
	'<div class="app-footer">Built with Python • Machine Learning • Scikit-learn • Streamlit</div>',
	unsafe_allow_html=True,
)
