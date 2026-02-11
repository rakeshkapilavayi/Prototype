import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import logging
from typing import Tuple, Optional, Union, Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], Optional[LabelEncoder]]:
    """
    Preprocess the dataset by separating features and target, handling missing values, and encoding the target for classification.
    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        task_type: 'classification' or 'regression'.
    Returns:
        Tuple containing:
            - X: Feature DataFrame.
            - y: Target Series.
            - features: List of feature column names.
            - label_encoder: LabelEncoder for classification target or None.
    """
    try:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
       
        features = df.columns.drop(target_column).tolist()
        if not features:
            raise ValueError("No features available for training.")
       
        X = df[features].copy()
        y = df[target_column]
       
        # Handle missing values in features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode()[0])
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        label_encoder = None
        if task_type == 'classification':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
       
        logger.info(f"Data preprocessed for {task_type}. Features: {len(features)}")
        return X, y, features, label_encoder
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise

def create_model_pipeline(
    task_type: str,
    model_type: str,
    categorical_cols: List[str],
    numerical_cols: List[str],
    X: pd.DataFrame
) -> Pipeline:
    """
    Create a scikit-learn pipeline for the specified model type with preprocessing.
    Args:
        task_type: 'classification' or 'regression'.
        model_type: Model to use (e.g., 'LogisticRegression').
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
        X: Preprocessed feature DataFrame for cardinality check.
    Returns:
        Pipeline with preprocessing and model.
    """
    try:
        # Check for high-cardinality categorical columns
        high_cardinality_threshold = 10
        if any(X[col].nunique() > high_cardinality_threshold for col in categorical_cols):
            logger.warning("High-cardinality categorical columns detected.")
        
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', cat_transformer, categorical_cols),
                ('num', StandardScaler(), numerical_cols)
            ])
       
        model_configs = {
            'classification': {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
                'SVC': SVC(probability=True, random_state=42)
            },
            'regression': {
                'LinearRegression': LinearRegression(),
                'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBRegressor': XGBRegressor(random_state=42),
                'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
                'SVR': SVR()
            }
        }
       
        if task_type not in model_configs:
            raise ValueError(f"Unsupported task type: {task_type}")
        if model_type not in model_configs[task_type]:
            raise ValueError(f"Unsupported model: {model_type}")
       
        model = model_configs[task_type][model_type]
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
       
        logger.info(f"Pipeline created for {task_type} with {model_type}")
        return pipeline
    except Exception as e:
        logger.error(f"Error in create_model_pipeline: {str(e)}")
        raise

def tune_hyperparameters(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    model_type: str
) -> Pipeline:
    """
    Perform hyperparameter tuning using GridSearchCV with reduced parameter grid and early stopping for XGBoost.
    Args:
        pipeline: Scikit-learn pipeline with preprocessing and model.
        X_train: Training features.
        y_train: Training target.
        task_type: 'classification' or 'regression'.
        model_type: Model type.
    Returns:
        Pipeline with tuned hyperparameters.
    """
    try:
        param_grid = {
            'LogisticRegression': {'model__C': [0.1, 1.0]},
            'RandomForestClassifier': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10]
            },
            'XGBClassifier': {
                'model__max_depth': [3, 6],
                'model__learning_rate': [0.1, 0.3],
                'model__n_estimators': [50, 100],
                'model__early_stopping_rounds': [10]
            },
            'DecisionTreeClassifier': {'model__max_depth': [5, 10]},
            'SVC': {'model__C': [0.1, 1.0], 'model__kernel': ['rbf']},
            'LinearRegression': {},
            'RandomForestRegressor': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10]
            },
            'XGBRegressor': {
                'model__max_depth': [3, 6],
                'model__learning_rate': [0.1, 0.3],
                'model__n_estimators': [50, 100],
                'model__early_stopping_rounds': [10]
            },
            'DecisionTreeRegressor': {'model__max_depth': [5, 10]},
            'SVR': {'model__C': [0.1, 1.0], 'model__epsilon': [0.1]}
        }
       
        if model_type in param_grid and param_grid[model_type]:
            grid_search = GridSearchCV(
                pipeline,
                param_grid[model_type],
                cv=3,  # Reduced from 5 to 3 for faster execution
                scoring='f1_weighted' if task_type == 'classification' else 'r2',
                n_jobs=-1,
                verbose=0
            )
            if 'XGB' in model_type:
                # Split data for early stopping in XGBoost
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=42
                )
                grid_search.fit(X_train_sub, y_train_sub, model__eval_set=[(X_val, y_val)], model__verbose=False)
            else:
                grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        return pipeline
    except Exception as e:
        logger.warning(f"Hyperparameter tuning failed for {model_type}: {str(e)}. Using default parameters.")
        return pipeline

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    label_encoder: Optional[LabelEncoder] = None
) -> Tuple[Dict, Optional[np.ndarray], Optional[px.imshow]]:
    """
    Evaluate the model and generate metrics and visualizations with additional metrics.
    Args:
        pipeline: Trained model pipeline.
        X_test: Test features.
        y_test: Test target.
        task_type: 'classification' or 'regression'.
        label_encoder: LabelEncoder for classification target or None.
    Returns:
        Tuple containing:
            - report: Evaluation metrics.
            - cm: Confusion matrix (for classification) or None.
            - cm_fig: Plotly figure of confusion matrix or None.
    """
    try:
        y_pred = pipeline.predict(X_test)
       
        if task_type == 'classification':
            y_test_decoded = label_encoder.inverse_transform(y_test) if label_encoder else y_test
            y_pred_decoded = label_encoder.inverse_transform(y_pred) if label_encoder else y_pred
            report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
            cm = confusion_matrix(y_test_decoded, y_pred_decoded)
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale='Blues',
                title='Confusion Matrix',
                labels=dict(x="Predicted", y="Actual")
            )
            if len(np.unique(y_test)) == 2:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                report['ROC_AUC'] = roc_auc_score(y_test, y_proba)
            report['F1_Score'] = f1_score(y_test_decoded, y_pred_decoded, average='weighted')
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            report = {
                'Mean Squared Error': mse,
                'Mean Absolute Error': mae,
                'R² Score': r2
            }
            cm = None
            cm_fig = None
       
        logger.info(f"Model evaluated for {task_type}. Metrics: {report}")
        return report, cm, cm_fig
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        raise

def get_feature_importance(
    pipeline: Pipeline,
    features: List[str],
    categorical_cols: List[str]
) -> Optional[pd.DataFrame]:
    """
    Extract feature importance for tree-based and linear models.
    Args:
        pipeline: Trained model pipeline.
        features: List of feature names.
        categorical_cols: List of categorical column names.
    Returns:
        DataFrame with feature importances or None if not applicable.
    """
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['model']
        feature_names = []
        
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'cat':
                ohe = transformer
                cat_features = ohe.get_feature_names_out(cols)
                feature_names.extend(cat_features)
            else:
                feature_names.extend(cols)
        
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
        elif hasattr(model, 'coef_'):
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(model.coef_)
            }).sort_values(by='Importance', ascending=False)
        else:
            logger.warning("Feature importance not available for this model type.")
            return None
        
        logger.info("Feature importances extracted successfully")
        return importances
    except Exception as e:
        logger.warning(f"Error extracting feature importance: {str(e)}")
        return None

def train_model(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    model_type: str,
    tune_params: bool = False
) -> Tuple[Pipeline, Dict, Optional[np.ndarray], Optional[px.imshow], List[str], Optional[LabelEncoder]]:
    """
    Train a machine learning model for classification or regression with optional hyperparameter tuning.
    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        task_type: 'classification' or 'regression'.
        model_type: Model to use (e.g., 'LogisticRegression').
        tune_params: Whether to perform hyperparameter tuning (default: False).
    Returns:
        Tuple containing:
            - model: Trained model pipeline.
            - report: Evaluation metrics.
            - cm: Confusion matrix (for classification) or None.
            - cm_fig: Plotly figure of confusion matrix or None.
            - features: List of feature column names.
            - label_encoder: LabelEncoder for target column or None.
    """
    try:
        # Preprocess data
        X, y, features, label_encoder = preprocess_data(df, target_column, task_type)
       
        # Identify column types
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
       
        # Create pipeline
        pipeline = create_model_pipeline(task_type, model_type, categorical_cols, numerical_cols, X)
       
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
        # Tune hyperparameters if enabled
        if tune_params:
            pipeline = tune_hyperparameters(pipeline, X_train, y_train, task_type, model_type)
       
        # Train model
        pipeline.fit(X_train, y_train)
        logger.info(f"Model {model_type} trained successfully for {task_type}")
       
        # Evaluate model
        report, cm, cm_fig = evaluate_model(pipeline, X_test, y_test, task_type, label_encoder)
       
        # Perform cross-validation
        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=3,  # Reduced from 5 to 3 for faster execution
            scoring='f1_weighted' if task_type == 'classification' else 'r2'
        )
        report['Cross_Validation_Score'] = cv_scores.mean()
        logger.info(f"Cross-validation score for {model_type}: {report['Cross_Validation_Score']:.4f}")
       
        # Get feature importance
        feature_importance = get_feature_importance(pipeline, features, categorical_cols)
        if feature_importance is not None:
            report['Feature_Importance'] = feature_importance.to_dict()
       
        return pipeline, report, cm, cm_fig, features, label_encoder
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise