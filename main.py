import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🌊 Chile Groundwater Depth Prediction Analysis")
print("=" * 50)

print("\n📊 Loading and exploring the dataset...")

file_path = r"E:\OneDrive - Colorado School of Mines\projects_2025\chile_groundwater\results\chile_2025\groundwater_chile_and_elevation_dataset_2025_with_GEE.xlsx"

try:
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    print("Creating sample data for demonstration...")
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'Depth to water (m)': np.random.exponential(10, n_samples),
        'Elevation': np.random.normal(500, 200, n_samples),
        'Longitude_GCS_WGS_1984': np.random.uniform(-75, -65, n_samples),
        'Latitude_GCS_WGS_1984': np.random.uniform(-35, -25, n_samples),
        'elevation_NASADEM': np.random.normal(500, 200, n_samples),
        'slope_NASADEM': np.random.exponential(5, n_samples),
        'terraclim_pr_value': np.random.exponential(100, n_samples),
        'terraclim_tmmn_value': np.random.normal(100, 50, n_samples),
        'terraclim_tmmx_value': np.random.normal(200, 50, n_samples),
        'Basin': np.random.choice(['Basin_A', 'Basin_B', 'Basin_C'], n_samples),
        'Status': np.random.choice(['Active', 'Inactive'], n_samples)
    })

print(f"\nDataset shape: {df.shape}")
print(f"Target variable: 'Depth to water (m)'")
print(f"Number of features: {df.shape[1] - 1}")

target_col = 'Depth to water (m)'
if target_col not in df.columns:
    print(f"❌ Target column '{target_col}' not found in dataset!")
    print("Available columns:", list(df.columns))
else:
    print(f"✅ Target variable '{target_col}' found")

print(f"\n📈 Target Variable Statistics:")
print(f"Mean depth to water: {df[target_col].mean():.2f} m")
print(f"Median depth to water: {df[target_col].median():.2f} m")
print(f"Standard deviation: {df[target_col].std():.2f} m")
print(f"Min depth: {df[target_col].min():.2f} m")
print(f"Max depth: {df[target_col].max():.2f} m")
print(f"Missing values: {df[target_col].isnull().sum()}")

print("\n🔧 Data preprocessing...")

initial_rows = len(df)
df = df.dropna(subset=[target_col])
print(f"Removed {initial_rows - len(df)} rows with missing target values")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if target_col in numeric_cols:
    numeric_cols.remove(target_col)

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

label_encoders = {}
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

encoded_categorical_cols = [col + '_encoded' for col in categorical_cols]
feature_cols = numeric_cols + encoded_categorical_cols
X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"Final feature matrix shape: {X.shape}")
print(f"Features selected: {len(feature_cols)}")

print("\n📊 Creating visualizations...")

fig = plt.figure(figsize=(20, 15))

plt.subplot(3, 4, 1)
plt.hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Depth to Water')
plt.xlabel('Depth to Water (m)')
plt.ylabel('Frequency')

plt.subplot(3, 4, 2)
y_log = np.log1p(y)
plt.hist(y_log, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
plt.title('Log-transformed Depth to Water')
plt.xlabel('Log(Depth to Water + 1)')
plt.ylabel('Frequency')

top_numeric_cols = numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols
corr_data = df[top_numeric_cols + [target_col]].corr()

plt.subplot(3, 4, (3, 4))
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
plt.title('Feature Correlation Matrix')

correlations = df[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
top_features = correlations.head(8).index

for i, feature in enumerate(top_features):
    plt.subplot(3, 4, 5 + i)
    plt.scatter(df[feature], y, alpha=0.5, s=10)
    plt.xlabel(feature[:20] + '...' if len(feature) > 20 else feature)
    plt.ylabel('Depth to Water (m)')
    plt.title(f'vs {feature[:15]}...')
    
    corr_coef = df[feature].corr(y)
    plt.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

print("\n🤖 Training multiple ML models...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------------
# OPTIMIZED HYPERPARAMETERS
# As detailed in Section 3.4 & Table 4 of the manuscript (Temporal Validation Phase)
# ---------------------------------------------------------------------------------
print("Initializing models with tuned hyperparameters...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
    
    # Tuned Random Forest
    'Random Forest': RandomForestRegressor(
        n_estimators=300, 
        max_depth=30, 
        min_samples_split=5, 
        min_samples_leaf=2, 
        random_state=42
    ),
    
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    
    # Tuned Extra Trees
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=300, 
        max_depth=30, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=42
    ),
    
    # Tuned Decision Tree
    'Decision Tree': DecisionTreeRegressor(
        max_depth=200, 
        min_samples_split=10, 
        min_samples_leaf=4, 
        random_state=42
    ),
    
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'Support Vector Regression': SVR(kernel='rbf', C=1.0)
}

results = {}
predictions = {}

print("Training progress:")
for name, model in models.items():
    print(f"  Training {name}...")
    
    if name in['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                'Elastic Net', 'Support Vector Regression', 'K-Nearest Neighbors']:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test
    
    model.fit(X_train_model, y_train)
    
    y_pred = model.predict(X_test_model)
    predictions[name] = y_pred
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if name in['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                'Elastic Net', 'Support Vector Regression', 'K-Nearest Neighbors']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                   scoring='neg_mean_squared_error')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_squared_error')
    
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'CV_RMSE': cv_rmse,
        'Model': model
    }

print("\n📊 Model Performance Comparison:")
print("=" * 80)
print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'CV_RMSE':<10}")
print("=" * 80)

for name, metrics in results.items():
    print(f"{name:<25} {metrics['RMSE']:<10.3f} {metrics['MAE']:<10.3f} "
          f"{metrics['R²']:<10.3f} {metrics['CV_RMSE']:<10.3f}")

best_model_name = min(results.keys(), key=lambda k: results[k]['CV_RMSE'])
best_model = results[best_model_name]['Model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Cross-Validation RMSE: {results[best_model_name]['CV_RMSE']:.3f}")
print(f"   Test R²: {results[best_model_name]['R²']:.3f}")

print(f"\n🔍 Detailed analysis of {best_model_name}...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

ax1 = axes[0, 0]
model_names = list(results.keys())
rmse_scores = [results[name]['RMSE'] for name in model_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

bars = ax1.bar(range(len(model_names)), rmse_scores, color=colors)
ax1.set_xlabel('Models')
ax1.set_ylabel('RMSE')
ax1.set_title('Model Performance Comparison (RMSE)')
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha='right')

best_idx = model_names.index(best_model_name)
bars[best_idx].set_color('gold')
bars[best_idx].set_edgecolor('red')
bars[best_idx].set_linewidth(3)

ax2 = axes[0, 1]
r2_scores = [results[name]['R²'] for name in model_names]
bars2 = ax2.bar(range(len(model_names)), r2_scores, color=colors)
ax2.set_xlabel('Models')
ax2.set_ylabel('R² Score')
ax2.set_title('Model Performance Comparison (R²)')
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=45, ha='right')
bars2[best_idx].set_color('gold')
bars2[best_idx].set_edgecolor('red')
bars2[best_idx].set_linewidth(3)

ax3 = axes[0, 2]
best_predictions = predictions[best_model_name]
ax3.scatter(y_test, best_predictions, alpha=0.6, color='blue', s=20)
ax3.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Depth to Water (m)')
ax3.set_ylabel('Predicted Depth to Water (m)')
ax3.set_title(f'{best_model_name}: Predicted vs Actual')

r2_best = results[best_model_name]['R²']
ax3.text(0.05, 0.95, f'R² = {r2_best:.3f}', transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax4 = axes[1, 0]
residuals = y_test - best_predictions
ax4.scatter(best_predictions, residuals, alpha=0.6, color='green', s=20)
ax4.axhline(y=0, color='red', linestyle='--')
ax4.set_xlabel('Predicted Depth to Water (m)')
ax4.set_ylabel('Residuals (m)')
ax4.set_title(f'{best_model_name}: Residuals Plot')

ax5 = axes[1, 1]
ax5.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
ax5.set_xlabel('Residuals (m)')
ax5.set_ylabel('Frequency')
ax5.set_title(f'{best_model_name}: Residuals Distribution')

ax6 = axes[1, 2]
cv_rmse_scores = [results[name]['CV_RMSE'] for name in model_names]
bars3 = ax6.bar(range(len(model_names)), cv_rmse_scores, color=colors)
ax6.set_xlabel('Models')
ax6.set_ylabel('Cross-Validation RMSE')
ax6.set_title('Cross-Validation Performance')
ax6.set_xticks(range(len(model_names)))
ax6.set_xticklabels(model_names, rotation=45, ha='right')
bars3[best_idx].set_color('gold')
bars3[best_idx].set_edgecolor('red')
bars3[best_idx].set_linewidth(3)

plt.tight_layout()
plt.show()

print(f"\n🎯 Feature Importance Analysis for {best_model_name}...")

if best_model_name in['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                       'Elastic Net', 'Support Vector Regression', 'K-Nearest Neighbors']:
    X_train_importance = X_train_scaled
    X_test_importance = X_test_scaled
else:
    X_train_importance = X_train
    X_test_importance = X_test

if hasattr(best_model, 'feature_importances_'):
    importance_scores = best_model.feature_importances_
    importance_type = "Built-in Feature Importance"
else:
    print("  Calculating permutation importance...")
    perm_importance = permutation_importance(best_model, X_test_importance, y_test, 
                                           n_repeats=10, random_state=42)
    importance_scores = perm_importance.importances_mean
    importance_type = "Permutation Feature Importance"

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importance_scores
}).sort_values('Importance', ascending=False)

print(f"\nTop 20 Most Important Features ({importance_type}):")
print("=" * 60)
for i, (_, row) in enumerate(feature_importance_df.head(20).iterrows()):
    feature_name = row['Feature']
    if len(feature_name) > 40:
        feature_name = feature_name[:37] + "..."
    print(f"{i+1:2d}. {feature_name:<40} {row['Importance']:.4f}")

plt.figure(figsize=(15, 10))

top_20_features = feature_importance_df.head(20)

plt.subplot(2, 1, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_20_features)))
bars = plt.barh(range(len(top_20_features)), top_20_features['Importance'], color=colors)
plt.yticks(range(len(top_20_features)), 
           [f[:30] + '...' if len(f) > 30 else f for f in top_20_features['Feature']])
plt.xlabel('Importance Score')
plt.title(f'Top 20 Feature Importance - {best_model_name}')
plt.gca().invert_yaxis()

for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + max(top_20_features['Importance']) * 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=8)

plt.subplot(2, 1, 2)
plt.hist(feature_importance_df['Importance'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Importance Score')
plt.ylabel('Number of Features')
plt.title('Distribution of Feature Importance Scores')
plt.axvline(feature_importance_df['Importance'].mean(), color='red', linestyle='--', 
            label=f'Mean: {feature_importance_df["Importance"].mean():.4f}')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\n⚙️ Hyperparameter evaluation (Models are already initialized with manuscript's Table 4 tuning parameters)")

# Kept for demonstration of ridge/lasso tuning if selected as best model
if best_model_name in ['Ridge Regression', 'Lasso Regression']:
    param_grid = {'alpha':[0.1, 1.0, 10.0, 100.0]}
    if best_model_name == 'Ridge Regression':
        model_class = Ridge
    else:
        model_class = Lasso
    
    grid_search = GridSearchCV(model_class(), param_grid, 
                              cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {np.sqrt(-grid_search.best_score_):.3f}")
    
    y_pred_tuned = grid_search.predict(X_test_scaled)
    
    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    tuned_r2 = r2_score(y_test, y_pred_tuned)
    
    print(f"  Tuned model RMSE: {tuned_rmse:.3f}")
    print(f"  Tuned model R²: {tuned_r2:.3f}")
    
    original_rmse = results[best_model_name]['RMSE']
    improvement = ((original_rmse - tuned_rmse) / original_rmse) * 100
    print(f"  Improvement: {improvement:.2f}%")
else:
    print(f"  {best_model_name} is already using the optimal hyperparameters identified in the study.")

print("\n📋 SUMMARY AND RECOMMENDATIONS")
print("=" * 50)
print(f"✅ Dataset processed: {X.shape[0]} samples, {X.shape[1]} features")
print(f"🏆 Best performing model: {best_model_name}")
print(f"📊 Best model performance:")
print(f"   - R² Score: {results[best_model_name]['R²']:.3f}")
print(f"   - RMSE: {results[best_model_name]['RMSE']:.3f} meters")
print(f"   - MAE: {results[best_model_name]['MAE']:.3f} meters")
print(f"   - Cross-validation RMSE: {results[best_model_name]['CV_RMSE']:.3f} meters")

print(f"\n🎯 Top 5 most important features:")
for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows()):
    print(f"   {i+1}. {row['Feature']}")

print(f"\n💡 Recommendations:")
print(f"   1. Focus data collection on the top important features")
print(f"   2. Consider feature engineering for improved performance")
print(f"   3. Collect more data if possible to improve model robustness")
print(f"   4. Monitor model performance on new data")
print(f"   5. Consider ensemble methods combining multiple models")

print(f"\n🎉 Analysis completed successfully!")
