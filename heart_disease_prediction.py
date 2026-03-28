# ============================================================
#   HEART DISEASE PREDICTION SYSTEM
#   Minor Project - Machine Learning
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

def load_dataset(filepath=None):
    if filepath:
        df = pd.read_csv(filepath)
    else:
        print("[INFO] No file provided. Using sample data for demonstration.")
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            'age':      np.random.randint(29, 77, n),
            'sex':      np.random.randint(0, 2, n),
            'cp':       np.random.randint(0, 4, n),
            'trestbps': np.random.randint(94, 200, n),
            'chol':     np.random.randint(126, 564, n),
            'fbs':      np.random.randint(0, 2, n),
            'restecg':  np.random.randint(0, 3, n),
            'thalach':  np.random.randint(71, 202, n),
            'exang':    np.random.randint(0, 2, n),
            'oldpeak':  np.round(np.random.uniform(0, 6.2, n), 1),
            'slope':    np.random.randint(0, 3, n),
            'ca':       np.random.randint(0, 5, n),
            'thal':     np.random.randint(0, 4, n),
            'target':   np.random.randint(0, 2, n)
        })
    return df


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================

def perform_eda(df):
    print("\n" + "="*60)
    print("       EXPLORATORY DATA ANALYSIS")
    print("="*60)

    print(f"\n Dataset Shape: {df.shape}")
    print(f"\n Columns: {list(df.columns)}")
    print(f"\n First 5 rows:\n{df.head()}")
    print(f"\n Statistical Summary:\n{df.describe()}")
    print(f"\n Missing Values:\n{df.isnull().sum()}")
    print(f"\n Target Distribution:\n{df['target'].value_counts()}")
    print(f"   Percentage with heart disease: {df['target'].mean()*100:.1f}%")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Heart Disease - Exploratory Data Analysis', fontsize=16, fontweight='bold')

    target_counts = df['target'].value_counts()
    axes[0, 0].pie(target_counts, labels=['No Disease', 'Disease'],
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0, 0].set_title('Target Distribution')

    axes[0, 1].hist(df[df['target']==0]['age'], alpha=0.7, label='No Disease',
                    color='#2ecc71', bins=20)
    axes[0, 1].hist(df[df['target']==1]['age'], alpha=0.7, label='Disease',
                    color='#e74c3c', bins=20)
    axes[0, 1].set_title('Age Distribution')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].legend()

    gender_target = pd.crosstab(df['sex'], df['target'])
    gender_target.plot(kind='bar', ax=axes[0, 2], color=['#2ecc71', '#e74c3c'])
    axes[0, 2].set_title('Gender vs Heart Disease')
    axes[0, 2].set_xticklabels(['Female', 'Male'], rotation=0)
    axes[0, 2].legend(['No Disease', 'Disease'])

    cp_target = pd.crosstab(df['cp'], df['target'])
    cp_target.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'])
    axes[1, 0].set_title('Chest Pain Type vs Heart Disease')
    axes[1, 0].set_xticklabels(['Type 0', 'Type 1', 'Type 2', 'Type 3'], rotation=0)
    axes[1, 0].legend(['No Disease', 'Disease'])

    df.boxplot(column='chol', by='target', ax=axes[1, 1],
               boxprops=dict(color='navy'), medianprops=dict(color='red'))
    axes[1, 1].set_title('Cholesterol by Target')
    axes[1, 1].set_xlabel('Target (0=No Disease, 1=Disease)')
    axes[1, 1].set_ylabel('Cholesterol')

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=axes[1, 2], cmap='coolwarm', annot=False,
                fmt='.2f', linewidths=0.5)
    axes[1, 2].set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n EDA plots saved as 'eda_plots.png'")

    return df


# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================

def preprocess_data(df):
    print("\n" + "="*60)
    print("       DATA PREPROCESSING")
    print("="*60)

    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median())
        print("Missing values filled with median")
    else:
        print("No missing values found")

    X = df.drop('target', axis=1)
    y = df['target']
    y = (y > 0).astype(int)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ============================================================
# STEP 4: MODEL TRAINING & COMPARISON
# ============================================================

def train_and_compare_models(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("       MODEL TRAINING & COMPARISON")
    print("="*60)

    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'Support Vector Machine': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(probability=True, random_state=42))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
    }

    results = {}
    best_model = None
    best_accuracy = 0

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'pipeline': pipeline,
            'accuracy': acc,
            'roc_auc': roc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        print(f"\n {name}")
        print(f"   Accuracy  : {acc*100:.2f}%")
        print(f"   ROC-AUC   : {roc:.4f}")
        print(f"   CV Score  : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = name

    print(f"\n Best Model: {best_model} with {best_accuracy*100:.2f}% accuracy")
    return results, best_model


# ============================================================
# STEP 5: DETAILED EVALUATION
# ============================================================

def evaluate_best_model(results, best_model, y_test, feature_names):
    print("\n" + "="*60)
    print(f"   DETAILED EVALUATION: {best_model}")
    print("="*60)

    r = results[best_model]
    y_pred = r['y_pred']
    y_prob = r['y_prob']

    print(f"\n Classification Report:\n")
    print(classification_report(y_test, y_pred,
          target_names=['No Disease', 'Heart Disease']))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Best Model: {best_model}', fontsize=14, fontweight='bold')

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {auc:.3f}')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1].set_title('ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    model_names = list(results.keys())
    accuracies = [results[m]['accuracy']*100 for m in model_names]
    colors = ['#e74c3c' if m == best_model else '#3498db' for m in model_names]
    short_names = [m.replace(' ', '\n') for m in model_names]
    bars = axes[2].bar(short_names, accuracies, color=colors, edgecolor='white', linewidth=1.5)
    axes[2].set_title('Model Accuracy Comparison')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_ylim([0, 110])
    for bar, acc in zip(bars, accuracies):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n Evaluation plots saved as 'model_evaluation.png'")

    model_obj = r['pipeline'].named_steps['model']
    if hasattr(model_obj, 'feature_importances_'):
        importances = model_obj.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)

        plt.figure(figsize=(8, 6))
        plt.barh(feat_df['Feature'], feat_df['Importance'], color='#3498db')
        plt.title(f'Feature Importance - {best_model}', fontweight='bold')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Feature importance plot saved as 'feature_importance.png'")

    return r['pipeline']


# ============================================================
# STEP 6: HYPERPARAMETER TUNING
# ============================================================

def tune_best_model(X_train, y_train, best_model_name):
    print("\n" + "="*60)
    print("       HYPERPARAMETER TUNING")
    print("="*60)

    if 'Random Forest' in best_model_name:
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 5, 10],
            'model__min_samples_split': [2, 5]
        }
        base = Pipeline([('scaler', StandardScaler()),
                         ('model', RandomForestClassifier(random_state=42))])
    elif 'Gradient' in best_model_name:
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth': [3, 5]
        }
        base = Pipeline([('scaler', StandardScaler()),
                         ('model', GradientBoostingClassifier(random_state=42))])
    elif 'Logistic' in best_model_name:
        param_grid = {'model__C': [0.01, 0.1, 1, 10, 100]}
        base = Pipeline([('scaler', StandardScaler()),
                         ('model', LogisticRegression(random_state=42, max_iter=1000))])
    else:
        print(f"Tuning not configured for {best_model_name}. Skipping.")
        return None

    grid = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best CV Accuracy: {grid.best_score_*100:.2f}%")

    return grid.best_estimator_


# ============================================================
# STEP 7: PREDICT SINGLE PATIENT
# ============================================================

def predict_single_patient(model, feature_names):
    print("\n" + "="*60)
    print("       SINGLE PATIENT PREDICTION")
    print("="*60)

    sample_patient = {
        'age': 52,
        'sex': 1,
        'cp': 2,
        'trestbps': 140,
        'chol': 250,
        'fbs': 0,
        'restecg': 1,
        'thalach': 155,
        'exang': 0,
        'oldpeak': 1.2,
        'slope': 1,
        'ca': 1,
        'thal': 2
    }

    print(f"\n Patient Data:")
    for k, v in sample_patient.items():
        print(f"   {k:12s}: {v}")

    patient_df = pd.DataFrame([sample_patient])[feature_names]
    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0]

    print(f"\n Prediction Result:")
    print(f"   Diagnosis   : {'HEART DISEASE DETECTED' if prediction == 1 else 'NO HEART DISEASE'}")
    print(f"   Probability : No Disease = {probability[0]*100:.1f}% | Disease = {probability[1]*100:.1f}%")

    return prediction, probability


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("   HEART DISEASE PREDICTION SYSTEM")
    print("   Minor Project - Machine Learning")
    print("="*60)

    # Real dataset use cheyyataniki:
    # df = load_dataset('heart.csv')

    # Sample data tho test cheyyataniki:
    df = load_dataset('heart.csv')

    df = perform_eda(df)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    results, best_model = train_and_compare_models(X_train, X_test, y_train, y_test)
    best_pipeline = evaluate_best_model(results, best_model, y_test, feature_names)
    tuned_model = tune_best_model(X_train, y_train, best_model)
    final_model = tuned_model if tuned_model else best_pipeline
    predict_single_patient(final_model, feature_names)

    print("\n" + "="*60)
    print("   PROJECT COMPLETE!")
    print("="*60)