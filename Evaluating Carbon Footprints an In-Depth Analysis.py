#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("emissions_high_granularity.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


# Handle Duplicates
duplicates = df.duplicated().sum()
print(f'Duplicates: {duplicates}')


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for a more modern look
plt.style.use('seaborn')

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(14, 8))

# Plot the data with a thicker line and larger markers
ax.plot(df['year'], df['total_emissions_MtCO2e'], marker='o', linestyle='-', linewidth=2, markersize=8)

# Customize the plot
ax.set_title('Total Emissions Over the Years', fontsize=20, pad=20)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Total Emissions (MtCO2e)', fontsize=14)

# Add gridlines
ax.grid(True, linestyle='--', alpha=0.7)

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Add data labels
for x, y in zip(df['year'], df['total_emissions_MtCO2e']):
    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

# Add a trend line
z = np.polyfit(df['year'], df['total_emissions_MtCO2e'], 1)
p = np.poly1d(z)
ax.plot(df['year'], p(df['year']), "r--", alpha=0.8, label="Trend Line")

# Add legend
ax.legend(fontsize=12)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for a more modern look
plt.style.use('seaborn')

# Get the commodity counts and sort them
commodity_counts = df['commodity'].value_counts()
commodity_counts_sorted = commodity_counts.sort_values(ascending=False)

# Select top 5 commodities and group others
top_5 = commodity_counts_sorted.head(5)
others = pd.Series({'Others': commodity_counts_sorted.iloc[5:].sum()})
data = pd.concat([top_5, others])

# Create a color palette
colors = sns.color_palette('pastel')[0:6]

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(12, 8))

# Create the pie chart
wedges, texts, autotexts = ax.pie(data, 
                                  labels=data.index, 
                                  autopct='%1.1f%%',
                                  pctdistance=0.85,
                                  colors=colors,
                                  startangle=90,
                                  wedgeprops=dict(width=0.5, edgecolor='white'))

# Add a circle at the center to create a donut chart effect
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Customize the chart
plt.title('Commodity Distribution', fontsize=20, pad=20)
plt.axis('equal')

# Enhance the appearance of labels and percentages
for text in texts + autotexts:
    text.set_fontsize(12)

# Add a legend
ax.legend(wedges, data.index,
          title="Commodities",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout()
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for a more modern look
plt.style.use('seaborn')

# Define emission columns
emissions_columns = ['product_emissions_MtCO2', 'flaring_emissions_MtCO2', 'venting_emissions_MtCO2', 
                     'own_fuel_use_emissions_MtCO2', 'fugitive_methane_emissions_MtCO2e', 'total_operational_emissions_MtCO2e']

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 14))
axes = axes.flatten()

# Custom color palette
colors = sns.color_palette("husl", 6)

for i, col in enumerate(emissions_columns):
    # Create histogram with KDE
    sns.histplot(data=df, x=col, kde=True, ax=axes[i], color=colors[i], edgecolor='black', alpha=0.7)
    
    # Customize each subplot
    axes[i].set_title(f'Distribution of\n{col}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel(col.replace('_', ' ').title(), fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    
    # Add mean and median lines
    mean = df[col].mean()
    median = df[col].median()
    axes[i].axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    axes[i].axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    
    # Add legend
    axes[i].legend(fontsize=10)
    
    # Use scientific notation for x-axis if values are very large
    if df[col].max() > 1000:
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# Adjust layout
plt.tight_layout()

# Add a main title
fig.suptitle('Distribution of Various Emission Types', fontsize=20, y=1.02)

plt.show()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the style for a more modern look
plt.style.use('seaborn')

# Get the top 10 parent entities
parent_entity_counts = df['parent_entity'].value_counts().head(10)

# Create a color palette
colors = sns.color_palette("viridis", n_colors=10)

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(16, 10))

# Create the bar plot
bars = ax.bar(parent_entity_counts.index, parent_entity_counts.values, color=colors)

# Customize the plot
ax.set_title('Top 10 Parent Entity Distribution', fontsize=20, pad=20)
ax.set_xlabel('Parent Entity', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)

# Rotate x-axis labels and adjust their alignment
plt.xticks(rotation=45, ha='right', fontsize=12)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,}',
            ha='center', va='bottom', fontsize=10)

# Add a grid for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Calculate and display the percentage each entity represents
total = parent_entity_counts.sum()
for i, v in enumerate(parent_entity_counts.values):
    percentage = v / total * 100
    ax.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# In[13]:


# Data Preprocessing
df = pd.get_dummies(df, columns=['parent_entity', 'parent_type', 'reporting_entity', 'commodity', 'production_unit', 'source'], drop_first=True)


# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# Assuming df is your DataFrame

# Separate features and target
X = df.drop(columns=['year', 'total_emissions_MtCO2e'])
y = df['total_emissions_MtCO2e']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and model training
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler()),  # Scale features
    ('feature_selection', SelectKBest(f_regression, k=10)),  # Select top 10 features
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.2f}")

# Feature importance
feature_importance = pipeline.named_steps['model'].feature_importances_
feature_names = pipeline.named_steps['feature_selection'].get_feature_names_out()

# Sort feature importances in descending order
indices = np.argsort(feature_importance)[::-1]

# Print the feature ranking
print("\nFeature ranking:")
for f in range(len(feature_names)):
    print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], feature_importance[indices[f]]))

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame

# Separate features and target
X = df.drop(columns=['year', 'total_emissions_MtCO2e'])
y = df['total_emissions_MtCO2e']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'Neural Network': MLPRegressor(random_state=42)
}

# Create a pipeline
def create_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_regression, k=10)),
        ('model', model)
    ])

# Train and evaluate models
results = []
for name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    
    results.append({
        'Model': name,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'CV Score': cv_scores.mean()
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Visualize model performance
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R2', data=results_df)
plt.title('Model Performance Comparison (R2 Score)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature importance for the best model (assuming Random Forest performs best)
best_model = create_pipeline(RandomForestRegressor(n_estimators=100, random_state=42))
best_model.fit(X, y)

feature_importance = best_model.named_steps['model'].feature_importances_
feature_names = best_model.named_steps['feature_selection'].get_feature_names_out()

# Sort feature importances in descending order
indices = np.argsort(feature_importance)[::-1]

# Print the feature ranking
print("\nFeature ranking:")
for f in range(len(feature_names)):
    print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], feature_importance[indices[f]]))

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    evs = explained_variance_score(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'MSE': mse,
        'MAE': mae,
        'R-squared': r2,
        'RMSE': rmse,
        'Explained Variance': evs
    }
    
    return metrics

# Assuming you have y_test and y_pred from your model predictions
metrics = evaluate_model(y_test, y_pred, 'Random Forest')

# Create a DataFrame for the metrics
metrics_df = pd.DataFrame([metrics])

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Print the metrics
print(metrics_df.T)
print(f"\nCross-validation R2 scores: {cv_scores}")
print(f"Mean CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Visualize the metrics
plt.figure(figsize=(12, 6))
sns.barplot(x=metrics_df.columns[1:], y=metrics_df.iloc[0, 1:])
plt.title(f'Evaluation Metrics for {metrics["Model"]}')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter plot of predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

# Distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.tight_layout()
plt.show()

