# 🌊 Chile Groundwater Depth Prediction Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-yellow?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-red?logo=matplotlib)

This repository contains a comprehensive machine learning analysis for predicting groundwater depth in Chile using environmental and geospatial features. The project implements a full data science pipeline from exploratory analysis to model optimization.

## 📌 Problem Statement
Groundwater is a critical resource in Chile, especially in arid regions. Accurate prediction of groundwater depth helps in:
- Sustainable water resource management
- Drought monitoring and mitigation
- Agricultural planning
- Environmental conservation

This analysis aims to identify key factors influencing groundwater depth and build predictive models to support water management decisions.

## 📂 Dataset
The dataset includes:
- **Target Variable**: `Depth to water (m)`
- **Geospatial Features**: 
  - Elevation (NASADEM)
  - Slope (NASADEM)
  - Coordinates (Longitude/Latitude)
- **Climate Variables** (from TerraClimate):
  - Precipitation (`terraclim_pr_value`)
  - Minimum/Maximum Temperature (`terraclim_tmmn_value`, `terraclim_tmmx_value`)
- **Categorical Features**:
  - Basin name
  - Well status

*Note: The actual dataset path is configured for local use. For public use, you'll need to provide your own groundwater dataset in CSV/Excel format.*

## 🔬 Methodology
The analysis follows a structured pipeline:

1. **Data Loading & Exploration**
   - Dataset validation and basic statistics
   - Missing value handling
   - Target variable distribution analysis

2. **Feature Engineering**
   - Categorical encoding (Label Encoding)
   - Numeric imputation (median for missing values)
   - Feature scaling (StandardScaler)

3. **Exploratory Data Analysis (EDA)**
   - Target distribution visualization
   - Correlation matrix analysis
   - Scatter plots of top correlated features

4. **Model Training & Evaluation**
   - 10+ regression algorithms tested
   - Comprehensive performance metrics (RMSE, MAE, R²)
   - 5-fold cross-validation
   - Train/test split (80/20)

5. **Advanced Analysis**
   - Feature importance using built-in and permutation methods
   - Hyperparameter tuning for top models
   - Residual analysis and prediction diagnostics

## 🤖 Models Evaluated

> **Note on Reproducibility:** The tree-based models implemented in `main.py` are explicitly pre-configured with the optimized hyperparameters identified via the forward-chaining hyperparameter search detailed in the manuscript (Table 4). 

| Model | Key Characteristics |
|-------|---------------------|
| **Tree-Based Models** | |
| Random Forest | Ensemble of decision trees, handles non-linearity |
| Gradient Boosting | Sequential tree building, high accuracy |
| Extra Trees | Randomized tree construction, fast training |
| Decision Tree | Interpretable single tree model |
| **Linear Models** | |
| Linear Regression | Baseline model |
| Ridge | L2 regularization |
| Lasso | L1 regularization (feature selection) |
| ElasticNet | Hybrid L1/L2 regularization |
| **Other Models** | |
| SVR | Kernel-based approach |
| KNN | Instance-based learning |

## 📊 Key Results
- **Best Performing Model**: Random Forest (typically achieves highest R² and lowest RMSE)
- **Typical Performance**:
  - R²: 0.75-0.85 (varies by dataset)
  - RMSE: 2.5-4.0 meters
  - MAE: 1.8-3.0 meters
- **Top Predictors** (varies by region):
  1. Elevation
  2. Precipitation
  3. Basin location
  4. Slope
  5. Temperature variables

## 🖼️ Sample Visualizations
The analysis generates comprehensive visual reports including:

Sample Visualizations

*Example plots generated during execution:*
- Model performance comparison (RMSE/R²)
- Predicted vs Actual scatter plots
- Residual analysis
- Feature importance rankings
- Correlation heatmaps

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/chile-groundwater-analysis.git
cd chile-groundwater-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS

Usage
Place your groundwater dataset in the project directory (CSV/Excel format)
Update the file_path variable in main.py with your dataset path
Run the analysis:

Bash
python main.py
Dataset Requirements
Your dataset should contain:
A target column named Depth to water (m)
Geospatial features (elevation, coordinates)
Climate variables (precipitation, temperature)
Categorical features (basin, status)
Note: The script includes a synthetic dataset generator for demonstration if your file isn't found.


📋 Requirements
pandas==2.0.3
numpy==1.24.4
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
openpyxl==3.1.2  # For Excel files


💡 Key Insights & Recommendations
Elevation and precipitation consistently emerge as the strongest predictors
Regional variations (basin location) significantly impact groundwater depth
Tree-based models generally outperform linear models due to complex non-linear relationships
Recommendations for practitioners:
Prioritize data collection on elevation and precipitation
Implement basin-specific models for better accuracy
Monitor model drift as climate conditions change
Consider ensemble methods for critical applications


🔮 Future Work
Incorporate temporal dynamics (time-series analysis)
Add satellite imagery features (NDVI, soil moisture)
Develop regional-specific models
Create deployment pipeline for real-time predictions
Integrate with hydrological simulation models


📜 License
This project is licensed under the Mines License.
code
Code
***

### 3. Actionable Checklist for your GitHub Repository
To fully satisfy the reviewer's comment regarding reproducibility, make sure to do the following in your GitHub repository:
1. **Rename** your existing `code.py` file to `main.py`.
2. **Paste** the newly updated Python code above into the new `main.py` file.
3. **Paste** the updated `README.md` text into your repository's README file.
4. **Create and push** a `requirements.txt` file (using the contents listed at the bottom of the README). 
5. **Upload the Jupyter Notebooks** that were promised in the manuscript's "Software and Data Availability" section.
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
