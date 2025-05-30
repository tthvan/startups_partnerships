# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: col
#     language: python
#     name: python3
# ---

# %% [markdown] id="JOFHQM8s86AI"
# # Calculate distance from startups to company A

# %% id="8yEO_E_qWCgH"
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# %% colab={"base_uri": "https://localhost:8080/", "height": 895} id="DOd6f1id85Um" outputId="48eb0f64-0ecf-42dd-ba87-54e593003f4f"
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
df = pd.read_excel(r"C:\Users\Admin\Documents\Studio Code\startup_partnerships\Germany Startups Dataset.xlsx", sheet_name='Encoded')
df

# %% id="xJz-pRQ6-6_l"
compA = (50.1235823,8.5727836)
df['dist_A'] = df.apply(lambda row: geodesic((row['Latitude'], row['Longitude']), compA).kilometers, axis=1)
df['dist_A']

# %% id="T2ADVnpwCeiq"
df.to_excel('distance.xlsx', index=False)

# %% [markdown] id="BT66ia9yB6DL"
# # Correlation

# %% id="cxnfcFHP01NH"
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# %% colab={"base_uri": "https://localhost:8080/", "height": 461} id="6FbNaBiL04Vk" outputId="d8d5ebab-844e-4c86-f6f9-844ad23bde08"
df = pd.read_excel('data2_after_imputed.xlsx')
df

# %% id="TrycHKDU1M4B"
cate_col = df[['AdTech',
'Cybersecurity',
'Digital Health',
'Digital Chemistry',
'E-Commerce',
'Education',
'FinTech',
'InsurTech',
'LegalTech',
'Logistics',
'MediaTech',
'Mobility',
'SaaS',
'Smart Infrastructure',
'Smart Systems',
'Cross-industry',
'Artificial intelligence',
'Blockchain',
'Robotics',
'Virtual Reality',
'Hardware',
'Software Development',
'Data Analytics',
'Internet of Things',
'Partner',
'Financing',
'Talents',
'Mentoring',
'funding_no',
'Hub Berlin',
'Hub Cologne',
'Hub Dresden/Leipzig',
'Hub Dortmund',
'Hub Frankfurt/Darmstadt',
'Hub Hamburg',
'Hub Karlsruhe',
'Hub Mannheim/Ludwigshafen',
'Hub Munich',
'Hub Potsdam',
'Hub Nuremberg/Erlangen',
'Hub Stuttgart',
'Not part of the network yet',
'size_no',
'market_no',
'b2b',
'b2c',
'num_headquart',
'dist_bin_no',
'num_of_founder',
'num_of_female',
'percent_female',
'high_potential'
]]

# %% id="ux6gIkpo1ROn"
# Calculate the matrix of contingency coefficients
num_vars = cate_col.shape[1]

def cramers_v(var1, var2):
    # Step 1: Create the contingency table (crosstab) between the two variables
    crosstab = pd.crosstab(var1, var2, dropna=False)

    # Step 2: Run the chi-squared test
    chi2, _, _, _ = chi2_contingency(crosstab)

    # Step 3: Calculate the total observations
    n = crosstab.to_numpy().sum()

    # Step 4: Get the minimum dimension (rows or columns) and subtract 1
    min_dim = min(crosstab.shape) - 1

    # Step 5: Calculate Cramér’s V
    if min_dim == 0 or n == 0:
        return np.nan  # Avoid division by zero if the table is empty or has 1 category
    else:
        return np.sqrt(chi2 / (n * min_dim))


# %% id="cxARUtbC1UxB"
rows = []
for col1 in cate_col.columns:
    col = []
    for col2 in cate_col.columns:
        cramers = cramers_v(cate_col[col1], cate_col[col2])
        col.append(round(cramers, 2))
    rows.append(col)

cramers_results = pd.DataFrame(rows, columns=cate_col.columns, index=cate_col.columns)
cramers_results

df1=pd.DataFrame(cramers_results,columns=cate_col.columns,index=cate_col.columns)


# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="gKH_JOnK04Sx" outputId="b4c554ac-0101-4894-e720-05a51dd5a038"
selected_vars = ['percent_female', 'Hub Karlsruhe', 'Internet of Things',
                 'funding_no', 'Artificial intelligence', 'Smart Systems',
                 'b2c', 'Data Analytics', 'dist_bin_no', 'high_potential']

# Filter the Cramér's V results to only include the specified variables
filtered_cramers_results = cramers_results.loc[selected_vars, selected_vars]

# Display the filtered DataFrame
filtered_cramers_results

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="DiiCcepU1YXo" outputId="8794d2d7-294a-4f0a-e3f9-82de4b14d5c9"
#Map Cramer's V heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(df1, annot=True, annot_kws={"size": 8}, fmt=".2f", linewidths=0.5)
plt.title("Cramér's V Heatmap", fontsize=18)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %% [markdown] id="CQaKrxXO2IUu"
# # Contingency Matrix

# %% colab={"base_uri": "https://localhost:8080/", "height": 750} id="ApWzTyDq1dNq" outputId="3e043e5e-73ce-4328-ec60-c3945bcb7ae1"
contingency_matrix = np.zeros((num_vars, num_vars))

for i, var1 in enumerate(cate_col.columns):
    for j, var2 in enumerate(cate_col.columns):
        if i != j:  # Exclude the same variables
            contingency_table = pd.crosstab(cate_col[var1], cate_col[var2])
            chi2, _, _, _ = chi2_contingency(contingency_table)
            contingency_coefficient = np.sqrt(chi2 / (chi2 + len(cate_col)))  # Calculate the contingency coefficient
            contingency_matrix[i, j] = contingency_coefficient

# Create a DataFrame to store the results
contingency_df = pd.DataFrame(contingency_matrix, columns=cate_col.columns, index=cate_col.columns)

plt.figure(figsize=(num_vars // 2, num_vars // 2))

# Create the heatmap using the contingency_df matrix
sns.heatmap(contingency_df, annot=True, fmt=".2f", linewidths=.5)

plt.title("Contingency Coefficient Heatmap")
plt.show()

# %% [markdown] id="r5y6quuh2TK_"
# # Feature Selection

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %% colab={"base_uri": "https://localhost:8080/", "height": 461} id="lAAtvWwzbCCA" outputId="98eb8a17-30e9-4c2d-8dbf-b820a1da35b9"
df = pd.read_excel(r"C:\Users\Admin\Documents\Studio Code\startup_partnerships\data2_after_imputed.xlsx")
df

# %% id="p4sUx8aV2Ou1"
from sklearn.feature_selection import chi2
X = df.drop(columns=['high_potential'])
y = df['high_potential']

#Use chi-squared to examine the association of
chi_scores, p_values = chi2(X, y)
chi2_df = pd.DataFrame({'Feature': X.columns, 'Chi2': chi_scores, 'P-value': p_values})
chi2_df = chi2_df.sort_values(by=['P-value'], ascending=True)

chi_values = pd.Series(chi_scores, index=X.columns)
chi_values.sort_values(ascending=True, inplace=True)


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="YBQ_PkBjQ7ae" outputId="2f79b550-cd47-464c-c37d-35c4094b4099"
chi2_df

# %% colab={"base_uri": "https://localhost:8080/", "height": 872} id="6L-OEFzs29gz" outputId="1492b1a0-975d-45a6-f4f2-daad739c9c18"
plt.figure(figsize=(10, 10))
chi_values.plot.barh()
plt.xlabel('Chi-squared Value')
plt.ylabel('Features')
plt.title('Chi-squared Test Results')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 527} id="_AUYGxSD3E0E" outputId="8dce564d-0811-409d-a0f3-27150405e87e"
df_feature_selection = df[['percent_female', 'Internet of Things','funding_no',
                           'Artificial intelligence','Smart Systems', 'Hub Karlsruhe',
                           'b2c', 'Data Analytics', 'dist_bin_no' ]]

df_feature_selection['high_potential'] = df['high_potential']

df_feature_selection


# %% [markdown] id="oCW0JJJdXHAy"
# # PCA

# %% colab={"base_uri": "https://localhost:8080/"} id="mL37zhU0XJ19" outputId="271a7efb-3de2-4ac2-f105-053fc60a9110"
#2 PCA
from sklearn.decomposition import PCA

# Select the correlated feature pairs for PCA
features_to_pca_1 = df_feature_selection[['Internet of Things', 'Smart Systems']]
features_to_pca_2 = df_feature_selection[['Data Analytics', 'Artificial intelligence']]


# Keep the other features (excluding the correlated ones and the target variable)
other_features = df_feature_selection.drop(columns=['Internet of Things', 'Smart Systems', 'Data Analytics', 'Artificial intelligence', 'high_potential'])

# Apply PCA to the first pair of features
pca_1 = PCA(n_components=1)
principal_components_1 = pca_1.fit_transform(features_to_pca_1)
pca_df_1 = pd.DataFrame(principal_components_1, columns=['IoT_SmartSystems_PC1'])

# Apply PCA to the second pair of features
pca_2 = PCA(n_components=1)
principal_components_2 = pca_2.fit_transform(features_to_pca_2)
pca_df_2 = pd.DataFrame(principal_components_2, columns=['AI_DataAnalytics_PC2'])


# Combine PCA components, other features, and target variable into the final DataFrame
df_final = pd.concat([other_features, pca_df_1, pca_df_2, df_feature_selection[['high_potential']]], axis=1)

# Display the resulting DataFrame
print(df_final.head())


# %% id="0QhesTzwX0KR"
df_feature_selection = df_final

# %% [markdown]
# # NEGATIVE LABELLING (Cluster)

# %%
#update 29.05.2025 (reviewing)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# %%
#Scale features 
features = [col for col in df_feature_selection.columns if col != 'high_potential']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feature_selection[features])
X_scaled_df = pd.DataFrame(X_scaled, columns=features)


# %%
#Clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df_feature_selection['cluster'] = kmeans.fit_predict(X_scaled)

partner_clusters = df_feature_selection[df_feature_selection['high_potential'] == 1]['cluster'].unique()
negative_pool = df_feature_selection[(df_feature_selection['high_potential'] == 0) & (~df_feature_selection['cluster'].isin(partner_clusters))]

len(negative_pool) #=66 negative samples only -> too little

# %%
#relax the condition: get clusters also for those that have little positive:negative sample ratio
cluster_counts = df_feature_selection.groupby('cluster')['high_potential'].value_counts().unstack().fillna(0)
cluster_counts['positive_ratio'] = cluster_counts[1] / (cluster_counts[0] + cluster_counts[1])

# Keep clusters where most samples are NOT positive
safe_clusters = cluster_counts[cluster_counts['positive_ratio'] < 0.03].index

negative_pool = df_feature_selection[(df_feature_selection['high_potential'] == 0) & (df_feature_selection['cluster'].isin(safe_clusters))]
len(negative_pool)


# %%
positive_samples = df_feature_selection[df_feature_selection['high_potential'] == 1]
balanced_df = pd.concat([positive_samples, negative_pool]).reset_index(drop=True)
balanced_df

# %%
#Visualize clusters
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

df['TSNE1'] = X_tsne[:, 0]
df['TSNE2'] = X_tsne[:, 1]

plt.figure(figsize=(6,6))
sns.scatterplot(data=df, x='TSNE1', y='TSNE2', hue='high_potential', palette='Set1')
plt.title("t-SNE Visualization of Partner vs Non-Partner")

plt.show()

# %% [markdown] id="HzoR6Im2F8-r"
# # MODELLING

# %% id="xRDSbbxATiLH"
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, make_scorer, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve

# %%
X = balanced_df.drop(columns=['high_potential', 'cluster'])  # drop cluster and target
y = balanced_df['high_potential']

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Training set:\n", y_train.value_counts())
print("Test set:\n", y_test.value_counts())


# %% [markdown] id="jJde6o_oTkCL"
# ## NO SMOTE

# %% [markdown] id="rMB61HWsUBlQ"
# ### Logistic Regression

# %% id="oeh_b5iQUAEr"
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)
predicted_probs = model_lr.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 1
y_pred = model_lr.predict(X_test)
lr_score = accuracy_score(y_test, y_pred)


# %% colab={"base_uri": "https://localhost:8080/"} id="o5GOmR7RUHf_" outputId="284ed25f-99bc-4c5b-f0ec-08f733d9c273"
report = classification_report(y_test,y_pred, target_names=['Class 0', 'Class 1'])
class1_metrics = report.split('\n\n')[1]
print(class1_metrics)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="iLxr86u8UJcO" outputId="084dcdce-ed8b-444b-ac25-bd33bf08dc4c"
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
class1_metrics = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy'],
    'Logistic Regression': [precision, recall, f1_score, support, lr_score]
})

class1_metrics

# %%
#check data leakage
overlap = set(X_train.index).intersection(set(X_test.index))
overlap #nothing means no overlap

# %%
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(max_depth=3, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

metrics = {
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy']
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_test, y_pred, pos_label=1, average='binary'
    )
    acc = accuracy_score(y_test, y_pred)

    metrics[name] = [precision, recall, f1_score, support, acc]

class1_metrics = pd.DataFrame(metrics)
class1_metrics

# %%
#test using kfold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5, random_state=42),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True))
    ])
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    print(f"{name}: mean precision = {scores.mean():.3f}, std = {scores.std():.3f}")


# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="pfD6Qdg6x0ja" outputId="2c591602-7e04-478c-ede6-0fcc9d715b23"
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Logistic Regression Model")
plt.show()

# %% [markdown] id="zthLiTk3YFE3"
# # Modelling (with resampling SMOTE)

# %%
from imblearn.over_sampling import SMOTE
balanced_df_clean = balanced_df.drop_duplicates(subset=balanced_df.columns.difference(['high_potential']))

X = balanced_df_clean.drop(columns=['high_potential'])
y = balanced_df_clean['high_potential']

# Separate positives and negatives
positives = balanced_df_clean[balanced_df_clean['high_potential'] == 1]
negatives = balanced_df_clean[balanced_df_clean['high_potential'] == 0]

# Set desired number of positives and negatives for test set
n_pos_test = 15
n_neg_test = 35

# Sample test set
test_pos = positives.sample(n=n_pos_test, random_state=42)
test_neg = negatives.sample(n=n_neg_test, random_state=42)

df_test = pd.concat([test_pos, test_neg])
df_train = balanced_df_clean.drop(df_test.index)

# Rebuild X/y
X_train = df_train.drop(columns=['high_potential'])
y_train = df_train['high_potential']
X_test = df_test.drop(columns=['high_potential'])
y_test = df_test['high_potential']


#Apply SMOTE on training set 
sm = SMOTE(sampling_strategy='minority', random_state=42)
oversampled_X, oversampled_Y = sm.fit_resample(
    df_train.drop(columns=['high_potential']),
    df_train['high_potential']
)

oversampled = pd.concat([
    pd.DataFrame(oversampled_Y, columns=['high_potential']),
    pd.DataFrame(oversampled_X, columns=df_train.drop(columns=['high_potential']).columns)
], axis=1).reset_index(drop=True)

# Check for leakage
X_test_rows = X_test.reset_index(drop=True).copy()
X_oversampled = oversampled.drop(columns=['high_potential'])

leakage_rows = pd.merge(X_test_rows, X_oversampled, how='inner')
print("Leakage", len(leakage_rows))

# %%
oversampled_Y.value_counts()

# %%
print(len(X_train),len(X_test))

# %%
y_test.value_counts()

# %% [markdown] id="zHuw5KuZZP3u"
# ### Logistic Regression

# %%
# Train logistic regression on oversampled data
model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(oversampled_X, oversampled_Y)

# Predict on the *original* test set
predicted_probs = model_lr.predict_proba(X_test)[:, 1]  # Probabilities for class 1
y_pred = model_lr.predict(X_test)

# Accuracy
lr_score = accuracy_score(y_test, y_pred)
lr_score


# %% id="8TVUGGLQZPeg"
#OLD - SKIP
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)
predicted_probs = model_lr.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 0
y_pred = model_lr.predict(X_test)
lr_score = accuracy_score(y_test, y_pred)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="L-IG0LSZY2aE" outputId="98c60a25-3f10-4e6e-a507-938cf7b266a6"
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
class1_metrics = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy'],
    'Logistic Regression': [precision, recall, f1_score, support, lr_score]
})

class1_metrics

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(random_state=42)
model.fit(oversampled.drop(columns=['high_potential']), oversampled['high_potential'])

y_pred = model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred, digits=3))

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="f58BSiG9yD18" outputId="5b229838-5a50-4ef3-ed43-6e06f7c8bda4"
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Logistic Regression Model")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 268} id="6KQRp2_xhOdN" outputId="366864ed-c02b-4190-8e1a-992413daff87"
0.# Get coefficients
coefficients = model_lr.coef_[0]  # Get the array of coefficients for the model
feature_names = X_train.columns  # Assuming X_train is a DataFrame
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})

coef_df

# %% colab={"base_uri": "https://localhost:8080/", "height": 268} id="QBxXgv7RlWNA" outputId="172a140b-29b5-4749-f82f-69e0d4cac49f"
coef_df['coefficient_abs'] = coef_df['Coefficient'].abs()

# Sort the DataFrame by the absolute value of the coefficients
coef_df_sorted = coef_df.sort_values(by='coefficient_abs', ascending=False)

coef_df_sorted

# %%
corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find highly correlated pairs
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_train = X_train.drop(columns=to_drop)
X_train


# %%
import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)
model_sm = sm.Logit(y_train, X_train_sm).fit()
print(model_sm.summary())


# %%
print(X_train.info())


# %% colab={"base_uri": "https://localhost:8080/", "height": 352} id="AABq33UgckjG" outputId="b7d80537-2c51-45ac-b205-e132d8699d46"
import statsmodels.api as sm
import numpy as np

X_train_sm = sm.add_constant(X_train)  
model_sm = sm.Logit(y_train, X_train_sm).fit()

# Get results
coefficients = model_sm.params
standard_errors = model_sm.bse
p_values = model_sm.pvalues
conf_int = model_sm.conf_int()
odds_ratios = np.exp(coefficients)

# Create a df
coef_df = pd.DataFrame({
    "Feature": coefficients.index,
    "Coefficient": coefficients.values,
    "Standard Error": standard_errors.values,
    "P-value": p_values.values,
    "Odds Ratio": odds_ratios.values,
    "95% CI Lower": conf_int[0].values,
    "95% CI Upper": conf_int[1].values
})

coef_df.reset_index(drop=True, inplace=True)
coef_df


# %% colab={"base_uri": "https://localhost:8080/", "height": 676} id="iozSiowKr-i1" outputId="a59bd8e6-b268-410e-da39-6ea941fcfe96"
# Predict probabilities for each record in the training set
y_train_pred_prob = model_sm.predict(X_train_sm)

X_train_with_probs = X_train.copy()
X_train_with_probs["Predicted_Probability"] = y_train_pred_prob
X_train_with_probs["Actual_Label"] = y_train.values  # actual

X_train_with_probs_sorted = X_train_with_probs.sort_values(by="Predicted_Probability", ascending=False).reset_index(drop=True)

# Display the top-ranked records
X_train_with_probs_sorted.tail(20)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="JbNfPssJBMBa" outputId="8e2c7f59-5b12-4633-c4ee-3cacb59f9173"
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Predict probabilities and classes
X_test_sm = sm.add_constant(X_test)  # Add intercept to test data
probabilities = model_sm.predict(X_test_sm)
predictions = (probabilities >= 0.5).astype(int)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

# Display Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print Metrics Summary
print("Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, probabilities)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()


# %% [markdown] id="YjlzMgetaEG7"
# # ROC - AUC

# %% colab={"base_uri": "https://localhost:8080/"} id="oKtspKe-9mYT" outputId="31a0e46f-54c0-4c13-eba8-560018494372"
log_prob = model_lr.predict_proba(X_test)
log_probs = log_prob[:, 1]
fpr, tpr, _ = roc_curve(y_test, log_probs)
log_auc=roc_auc_score(y_test, log_probs)
log_auc

# %% [markdown] id="cHeWtWG5aWK-"
# # Validation

# %% id="7wFzz28FaP_e"
from sklearn.model_selection import cross_val_score, KFold

# %% colab={"base_uri": "https://localhost:8080/", "height": 362} id="0xY2fLlPaZl4" outputId="5707ed56-5304-4a84-d370-c9bfa7b934c8"
#kfold
X = df_feature_selection.drop(columns=['high_potential'], axis=1)
y = df_feature_selection['high_potential']

k = 10
scores_lr = cross_val_score(model_lr, X, y, cv=k, scoring='roc_auc')


df_validate = pd.DataFrame({'Fold': range(1, k+1), 'LR': scores_lr}) #Make a df storing all
df_validate

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} id="Lz_yudMMaiBH" outputId="078ca2e2-6e4c-4237-8741-f06acf37cf44"
#Add the mean and standard dev. of each validating results to df
mean_row = df_validate[['LR']].mean()
std_row = df_validate[['LR']].std()
df_validate.loc['Mean'] = [None] + list(mean_row)  # None for the 'Fold' column
df_validate.loc['Std Dev'] = [None] + list(std_row)
df_validate

# %% colab={"base_uri": "https://localhost:8080/", "height": 362} id="0fUgJU7JeWXV" outputId="abacd015-e2a0-41a6-f757-7a8b7a2db3fa"
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd

# Define your features and target
X = df_feature_selection.drop(columns=['high_potential'], axis=1)
y = df_feature_selection['high_potential']

# Initialize your logistic regression model (assuming model_lr is defined)
k = 10
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # StratifiedKFold for balanced splits

# Cross-validation with AUC-ROC
scores_lr = cross_val_score(model_lr, X, y, cv=cv, scoring='roc_auc')

# Calculate AUC-ROC for each fold manually for validation purposes
roc_auc_values = []
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_lr.fit(X_train, y_train)  # Fit model on training data
    y_pred_proba = model_lr.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
    auc_score = roc_auc_score(y_test, y_pred_proba)  # Calculate AUC-ROC for this fold
    roc_auc_values.append(auc_score)

# Create a DataFrame to store the results
df_validate = pd.DataFrame({
    'Fold': range(1, k+1),
    'AUC-ROC (cross_val_score)': scores_lr,  # AUC-ROC from cross_val_score
    'AUC-ROC (manual)': roc_auc_values      # AUC-ROC calculated manually
})

df_validate


# %% colab={"base_uri": "https://localhost:8080/", "height": 394} id="RnLkvpeDa0YZ" outputId="cf7b21e6-ee3b-4ee1-be99-7711f09c19c6"
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd

X = df_feature_selection.drop(columns=['high_potential'], axis=1)
y = df_feature_selection['high_potential']

# Initialize your logistic regression model (assuming model_lr is defined)
k = 10
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # StratifiedKFold for balanced splits

# Cross-validation with AUC-ROC
scores_lr = cross_val_score(model_lr, X, y, cv=cv, scoring='roc_auc')

# Calculate AUC-ROC for each fold manually for validation purposes
roc_auc_values = []
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_lr.fit(X_train, y_train)  # Fit model on training data
    y_pred_proba = model_lr.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
    auc_score = roc_auc_score(y_test, y_pred_proba)  # Calculate AUC-ROC for this fold
    roc_auc_values.append(auc_score)

# Create a DataFrame to store the results
df_validate = pd.DataFrame({
    'Fold': range(1, k+1),
    'AUC-ROC (cross_val_score)': scores_lr,  # AUC-ROC from cross_val_score
    'AUC-ROC (manual)': roc_auc_values      # AUC-ROC calculated manually
})

# Calculate the mean AUC-ROC for both methods
mean_auc_cross_val = scores_lr.mean()
mean_auc_manual = pd.Series(roc_auc_values).mean()

# Add the mean values to the DataFrame
df_validate.loc['Mean'] = ['Mean', mean_auc_cross_val, mean_auc_manual]

# Display the results
df_validate



# %% [markdown] id="xYQ-8Q2UGofh"
# # Graphs (discussion)

# %% colab={"base_uri": "https://localhost:8080/", "height": 895} id="q5uIFRIPsyuK" outputId="d1855af4-3ecc-4979-c032-9017660a5c48"
df_ori = pd.read_excel('Germany Startups Dataset.xlsx', sheet_name='Encoded')
df_ori

# %% colab={"base_uri": "https://localhost:8080/", "height": 461} id="ZL2ZPcYrtOBq" outputId="0d90988b-7440-4ccc-e579-b3842123a829"
df['lat'] = df_ori['Latitude']
df['lng'] = df_ori['Longitude']
df

# %% colab={"base_uri": "https://localhost:8080/", "height": 582} id="BXdgOpZjuoEJ" outputId="e7e21126-1ce7-4d67-863f-89aec4d569d7"
df_feature_selection = df[['percent_female', 'Internet of Things','funding_no',
                           'Artificial intelligence','Smart Systems', 'Hub Karlsruhe',
                           'b2c', 'Data Analytics', 'dist_bin_no','lat','lng' ]]

df_feature_selection['high_potential'] = df['high_potential']

df_feature_selection


# %% id="QTNI1TWayHRv"
#2 PCA
from sklearn.decomposition import PCA

# Select the correlated feature pairs for PCA
features_to_pca_1 = df_feature_selection[['Internet of Things', 'Smart Systems']]
features_to_pca_2 = df_feature_selection[['Data Analytics', 'Artificial intelligence']]


# Keep the other features (excluding the correlated ones and the target variable)
other_features = df_feature_selection.drop(columns=['Internet of Things', 'Smart Systems', 'Data Analytics', 'Artificial intelligence', 'high_potential'])

# Apply PCA to the first pair of features
pca_1 = PCA(n_components=1)
principal_components_1 = pca_1.fit_transform(features_to_pca_1)
pca_df_1 = pd.DataFrame(principal_components_1, columns=['IoT_SmartSystems_PC1'])

# Apply PCA to the second pair of features
pca_2 = PCA(n_components=1)
principal_components_2 = pca_2.fit_transform(features_to_pca_2)
pca_df_2 = pd.DataFrame(principal_components_2, columns=['AI_DataAnalytics_PC2'])


# Combine PCA components, other features, and target variable into the final DataFrame
df_final = pd.concat([other_features, pca_df_1, pca_df_2, df_feature_selection[['high_potential']]], axis=1)

# Display the resulting DataFrame

df_feature_selection = df_final

# %% colab={"base_uri": "https://localhost:8080/"} id="yfpiQvSpuVsg" outputId="1b36d537-93a7-4db0-e155-a7291af26d97"
from sklearn.model_selection import train_test_split

# Stratified split to maintain proportions of the target variable
df_train, df_test = train_test_split(df_feature_selection,
                                     train_size=0.7,
                                     test_size=0.3,
                                     random_state=100,
                                     stratify=df_feature_selection['high_potential'])

# Separate target variable and features for train and test sets
y_train = df_train['high_potential']
X_train = df_train.drop(columns='high_potential')
y_test = df_test['high_potential']
X_test = df_test.drop(columns='high_potential')

# Check the distribution in train and test sets
print("Training set class distribution:\n", y_train.value_counts())
print("Test set class distribution:\n", y_test.value_counts())


# %% colab={"base_uri": "https://localhost:8080/", "height": 530} id="8Zx_598-MGZ2" outputId="dd2c5cd8-0449-4e5a-83dc-6b7e136e13f0"
from imblearn.over_sampling import SMOTE
# Resampling the minority class. The strategy can be changed as required.
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(df_train.drop('high_potential', axis=1), df_train['high_potential'])
oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
oversampled

# %% id="oB7ECExLwOd8"
oversampled.to_excel('SMOTE_wlatlng.xlsx', index=False)

# %% colab={"base_uri": "https://localhost:8080/", "height": 462} id="Bb3jHM-kenO7" outputId="8af32618-1be4-4a02-c49b-24d99141c0a5"
# Ensure category mapping is applied correctly
category_mapping = {
    1: '0-20 km',
    2: '20-100 km',
    3: '100-200 km',
    4: '>200 km'
}

# Update category names in the dataset
oversampled['dist_bin_no'] = oversampled['dist_bin_no'].map(category_mapping)

category_order = ['0-20 km', '20-100 km', '100-200 km', '>200 km']
oversampled['dist_bin_no'] = pd.Categorical(oversampled['dist_bin_no'], categories=category_order, ordered=True)

# Calculate proportions
proportions = (
    oversampled.groupby('dist_bin_no')['high_potential']
    .mean()
    .reset_index()
    .rename(columns={'high_potential': 'proportion'})
)

# Sort categories if needed (ensure the order is consistent)
proportions = proportions.sort_values(by='dist_bin_no')

# Create Bar Chart
plt.figure(figsize=(8, 5))
bars = plt.bar(proportions['dist_bin_no'], proportions['proportion'] * 100, color='#d63627')
plt.xlabel('Distance Categories')
plt.ylabel('Percentage of High-Potential Startups in each Distance Bin')
plt.ylim(0, 100)  # Set y-axis limit to percentage range
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# Annotate bars with percentages
for bar in bars:
    height = bar.get_height()
    if not pd.isna(height):  # Skip NaN values
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 462} id="7VNLeA4sHT5L" outputId="03e311f0-d345-4f5c-c9cc-d2b144133885"
# Ensure category mapping is applied correctly
category_mapping = {
    1: 'Pre-Seed',
    2: 'Seed',
    3: 'Early Stage',
    4: 'Growth Stage',
    5: 'Later Stage'
}
oversampled['funding_no'] = oversampled['funding_no'].map(category_mapping)

category_order = ['Pre-Seed', 'Seed', 'Early Stage', 'Growth Stage', 'Later Stage']
oversampled['funding_no'] = pd.Categorical(oversampled['funding_no'], categories=category_order, ordered=True)

# Calculate proportions
proportions = (
    oversampled.groupby('funding_no')['high_potential']
    .mean()
    .reset_index()
    .rename(columns={'high_potential': 'proportion'})
)

# Bar Chart
plt.figure(figsize=(8, 5))
bars = plt.bar(proportions['funding_no'], proportions['proportion'] * 100, color = '#d63627')
plt.xlabel('Funding Phase')
plt.ylabel('Percentage of High-Potential Startups in Each Funding Phase')
plt.ylim(0, 100)  # Set y-axis limit to percentage range
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate percentages
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# %% id="G1U_nTNG_4JR"
category_mapping = {
    1: 'Pre-Seed',
    2: 'Seed',
    3: 'Early Stage',
    4: 'Growth Stage',
    5: 'Later Stage'
}
oversampled['funding_no'] = oversampled['funding_no'].map(category_mapping)

category_order = ['Pre-Seed', 'Seed', 'Early Stage', 'Growth Stage', 'Later Stage']
oversampled['funding_no'] = pd.Categorical(oversampled['funding_no'], categories=category_order, ordered=True)

# Calculate proportions
proportions = (
    oversampled.groupby('funding_no')['high_potential']
    .mean()
    .reset_index()
    .rename(columns={'high_potential': 'proportion'})
)

# Bar Chart
plt.figure(figsize=(8, 5))
bars = plt.bar(proportions['funding_no'], proportions['proportion'] * 100, color = '#d63627')
plt.xlabel('Funding Phase')
plt.ylabel('Percentage of High-Potential Startups in Each Funding Phase')
plt.ylim(0, 100)  # Set y-axis limit to percentage range
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate percentages
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

