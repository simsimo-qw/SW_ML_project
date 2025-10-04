import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_wine_data(red_wine_path: str, white_wine_path: str) -> pd.DataFrame:
    
    try:
        #load datasets
        red_wine = pd.read_csv(red_wine_path, delimiter=';')
        white_wine = pd.read_csv(white_wine_path, delimiter=';')
        
        print(f"Loaded red wine dataset: {red_wine.shape}")
        print(f"Loaded white wine dataset: {white_wine.shape}")
        
        #add wine type feature
        red_wine['wine_type'] = 0  # Red wine
        white_wine['wine_type'] = 1  # White wine
        #combined data
        combined_data = pd.concat([red_wine, white_wine], ignore_index=True)
        
        print(f"Combined dataset shape: {combined_data.shape}")
        
        return combined_data
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def explore_data(data: pd.DataFrame) -> Dict:
    
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    
    print(f"\n1. Dataset Overview:")
    print(f"Shape: {data.shape}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    
    print(f"\n2. Data Types and Missing Values:")
    info_df = pd.DataFrame({
        'Data Type': data.dtypes,
        'Missing Values': data.isnull().sum(),
        'Missing %': (data.isnull().sum() / len(data) * 100).round(2)
    })
    print(info_df)
    
    
    print(f"\n3. Descriptive Statistics:")
    print(data.describe())
    
    # binary target 
    if 'quality_binary' not in data.columns:
        data['quality_binary'] = (data['quality'] >= 6).astype(int)
    
    # Target variable analysis
    print(f"\n4. Target Variable Analysis:")
    quality_dist = data['quality'].value_counts().sort_index()
    binary_dist = data['quality_binary'].value_counts()
    
    print("Original quality distribution:")
    print(quality_dist)
    print(f"\nBinary quality distribution:")
    print(f"Bad wines (quality < 6): {binary_dist[0]} ({binary_dist[0]/len(data)*100:.1f}%)")
    print(f"Good wines (quality >= 6): {binary_dist[1]} ({binary_dist[1]/len(data)*100:.1f}%)")
    
    # Feature correlations with target
    print(f"\n5. Feature Correlations with Target:")
    feature_cols = [col for col in data.columns if col not in ['quality', 'quality_binary']]
    correlations = data[feature_cols].corrwith(data['quality_binary']).abs().sort_values(ascending=False)
    
    print("Top correlated features (absolute correlation):")
    for feature, corr in correlations.head(10).items():
        print(f"{feature:25s}: {corr:.4f}")
    
    return {
        'shape': data.shape,
        'quality_distribution': quality_dist.to_dict(),
        'binary_distribution': binary_dist.to_dict(),
        'feature_correlations': correlations.to_dict()
    }

def preprocess_data(
    red_wine_path: str,
    white_wine_path: str,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    scaler_type: str = 'standard',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
   
    print("="*80)
    print("DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load data
    wine_data = load_wine_data(red_wine_path, white_wine_path)
    
    # Explore data
    exploration_results = explore_data(wine_data)
    
    # Create binary target
    wine_data['quality_binary'] = (wine_data['quality'] >= 6).astype(int)
    
    # Select features (exclude original quality and binary target for X)
    feature_columns = [col for col in wine_data.columns if col not in ['quality', 'quality_binary']]
    X = wine_data[feature_columns].values
    y = wine_data['quality_binary'].values
    
    print(f"\nFeature Preparation:")
    print(f"Features selected: {feature_columns}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"\nData Split:")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    print(f"\nClass Distribution:")
    print(f"Train - Bad: {np.sum(y_train == 0)}, Good: {np.sum(y_train == 1)}")
    print(f"Val   - Bad: {np.sum(y_val == 0)}, Good: {np.sum(y_val == 1)}")
    print(f"Test  - Bad: {np.sum(y_test == 0)}, Good: {np.sum(y_test == 1)}")
    
    # Feature scaling
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    print(f"\nScaling features using {scaler_type} scaler...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Feature scaling completed")
    print(f"Feature means after scaling: {X_train_scaled.mean(axis=0)[:5].round(3)}")
    print(f"Feature stds after scaling: {X_train_scaled.std(axis=0)[:5].round(3)}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_columns


def create_data_visualizations(wine_data: pd.DataFrame, save_path: Optional[str] = None) -> None:
  
    print("Creating data visualizations...")
    
    # Create binary target if not exists
    if 'quality_binary' not in wine_data.columns:
        wine_data['quality_binary'] = (wine_data['quality'] >= 6).astype(int)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Wine Quality Dataset Overview', fontsize=16, fontweight='bold')
    
    # Quality distribution
    axes[0, 0].hist(wine_data['quality'], bins=range(3, 11), alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Original Quality Distribution')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Count')
    
    # Binary quality distribution
    binary_counts = wine_data['quality_binary'].value_counts()
    axes[0, 1].bar(['Bad (<6)', 'Good (≥6)'], binary_counts.values, 
                   color=['red', 'green'], alpha=0.7)
    axes[0, 1].set_title('Binary Quality Distribution')
    axes[0, 1].set_ylabel('Count')
    
    # Wine type distribution
    if 'wine_type' in wine_data.columns:
        wine_counts = wine_data['wine_type'].value_counts()
        axes[1, 0].bar(['Red', 'White'], wine_counts.values, 
                       color=['darkred', 'gold'], alpha=0.7)
        axes[1, 0].set_title('Wine Type Distribution')
        axes[1, 0].set_ylabel('Count')
    
    # Feature importance (correlation with target)
    feature_cols = [col for col in wine_data.columns if col not in ['quality', 'quality_binary']]
    correlations = wine_data[feature_cols].corrwith(wine_data['quality_binary']).abs().sort_values(ascending=True)
    
    axes[1, 1].barh(range(len(correlations)), correlations.values)
    axes[1, 1].set_yticks(range(len(correlations)))
    axes[1, 1].set_yticklabels([name.replace('_', ' ').title() for name in correlations.index])
    axes[1, 1].set_title('Feature Importance (Abs. Correlation)')
    axes[1, 1].set_xlabel('Absolute Correlation with Quality')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/data_overview.png", dpi=300, bbox_inches='tight')
    
    plt.show()


def create_preprocessing_pipeline(
    red_wine_path: str,
    white_wine_path: str,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    scaler_type: str = 'standard',
    create_plots: bool = True,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
   
    
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(
        red_wine_path, white_wine_path, test_size, validation_size, scaler_type
    )
    
   
    if create_plots:
        
        wine_data = load_wine_data(red_wine_path, white_wine_path)
        create_data_visualizations(wine_data, save_path)
    
    
    print(f" Final dataset shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   Features: {len(feature_names)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


if __name__ == "__main__":
    # Test the preprocessing pipeline
    try:
        red_path = "../data/winequalityred.csv"
        white_path = "../data/winequalitywhite.csv"
        
        X_train, X_val, X_test, y_train, y_val, y_test, features = create_preprocessing_pipeline(
            red_path, white_path, create_plots=True
        )
        
        print("Preprocessing test completed successfully!")
        
    except FileNotFoundError:
        print("Test files not found. This is expected if running as test.")
    except Exception as e:
        print(f"Test failed: {e}")