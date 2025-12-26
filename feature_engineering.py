"""
Gaming Churn Prediction - Feature Engineering
==============================================
Advanced feature engineering for churn prediction models.
Creates interaction features, time-based features, and behavioral signals.

Author: Priyanka Rawat
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ChurnFeatureEngineering:
    """
    Feature engineering pipeline for churn prediction.
    
    Creates 25+ features from raw gaming data including:
    - Engagement metrics
    - Spending patterns
    - Social interaction features
    - Time-based features
    - Behavioral change indicators
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def create_engagement_features(self, df):
        """Create composite engagement scores."""
        print("Creating engagement features...")
        
        # Overall engagement score (0-100)
        df['engagement_score'] = (
            (df['session_frequency_per_week'] / 14) * 25 +  # Max 25 points
            (df['avg_session_duration_min'] / 120) * 25 +   # Max 25 points
            (df['total_playtime_hours'] / 200) * 25 +        # Max 25 points
            (df['achievement_count'] / 150) * 25             # Max 25 points
        )
        df['engagement_score'] = df['engagement_score'].clip(0, 100)
        
        # Play intensity (hours per day since registration)
        df['play_intensity'] = df['total_playtime_hours'] / (df['days_since_registration'] + 1)
        
        # Session consistency (how regular are play sessions)
        df['session_consistency'] = df['session_frequency_per_week'] / 7
        
        # Achievement velocity (achievements per hour played)
        df['achievement_velocity'] = df['achievement_count'] / (df['total_playtime_hours'] + 1)
        
        # Engagement trend (are they becoming more or less engaged?)
        # Higher value = more recent engagement
        df['recency_score'] = np.exp(-df['days_since_last_login'] / 7)
        
        return df
    
    def create_spending_features(self, df):
        """Create spending behavior features."""
        print("Creating spending features...")
        
        # Spending per hour played
        df['spending_per_hour'] = df['total_spending_usd'] / (df['total_playtime_hours'] + 1)
        
        # Spending per session
        total_sessions = df['session_frequency_per_week'] * (df['days_since_registration'] / 7)
        df['spending_per_session'] = df['total_spending_usd'] / (total_sessions + 1)
        
        # Purchase frequency (purchases per week)
        df['purchase_frequency'] = df['in_game_purchases_count'] / (df['days_since_registration'] / 7 + 1)
        
        # Average transaction value
        df['avg_transaction_value'] = df['total_spending_usd'] / (df['in_game_purchases_count'] + 1)
        
        # Spending indicator (binary: spent or not)
        df['is_spender'] = (df['total_spending_usd'] > 0).astype(int)
        
        # Spending tier (categorize spending levels)
        df['spending_tier'] = pd.cut(
            df['total_spending_usd'],
            bins=[-0.01, 0, 10, 50, 100, 1000],
            labels=['Non-Spender', 'Minnow', 'Dolphin', 'Whale', 'Mega-Whale']
        )
        
        return df
    
    def create_social_features(self, df):
        """Create social interaction features."""
        print("Creating social features...")
        
        # Social engagement score
        df['social_score'] = (
            np.log1p(df['friend_count']) * 50 +
            np.log1p(df['chat_messages_sent']) * 50
        )
        
        # Chat activity per friend (how social within friend group)
        df['chat_per_friend'] = df['chat_messages_sent'] / (df['friend_count'] + 1)
        
        # Social vs solo play ratio
        df['social_ratio'] = df['friend_count'] / (df['total_playtime_hours'] + 1)
        
        # Is socially active (binary)
        df['is_social'] = ((df['friend_count'] > 5) | (df['chat_messages_sent'] > 50)).astype(int)
        
        return df
    
    def create_progression_features(self, df):
        """Create game progression features."""
        print("Creating progression features...")
        
        # Leveling speed (level per hour played)
        df['leveling_speed'] = df['level_reached'] / (df['total_playtime_hours'] + 1)
        
        # Achievement completion rate
        df['achievement_completion_rate'] = df['achievement_count'] / 150  # Max 150 achievements
        
        # Progression score (composite of level and achievements)
        df['progression_score'] = (
            (df['level_reached'] / 100) * 60 +
            df['achievement_completion_rate'] * 40
        )
        
        # Are they progressing quickly? (top 25% leveling speed)
        df['fast_progressor'] = (df['leveling_speed'] > df['leveling_speed'].quantile(0.75)).astype(int)
        
        return df
    
    def create_time_features(self, df):
        """Create time-based features."""
        print("Creating time-based features...")
        
        # Player lifecycle stage
        df['lifecycle_stage'] = pd.cut(
            df['days_since_registration'],
            bins=[0, 7, 30, 90, 180, 365],
            labels=['New', 'Early', 'Established', 'Veteran', 'Loyal']
        )
        
        # Inactivity risk (exponential decay)
        df['inactivity_risk'] = 1 - np.exp(-df['days_since_last_login'] / 7)
        
        # Activity decay rate (how long since last login relative to total time)
        df['activity_decay'] = df['days_since_last_login'] / (df['days_since_registration'] + 1)
        
        # Days since last login buckets
        df['last_login_bucket'] = pd.cut(
            df['days_since_last_login'],
            bins=[-1, 1, 3, 7, 14, 30, 365],
            labels=['Today/Yesterday', 'Recent', 'Week', '2-Weeks', 'Month', 'Long-Inactive']
        )
        
        # Expected next login (based on frequency)
        df['expected_next_login'] = 7 / (df['session_frequency_per_week'] + 0.1)
        df['overdue_for_login'] = (df['days_since_last_login'] > df['expected_next_login']).astype(int)
        
        return df
    
    def create_behavioral_change_features(self, df):
        """Create features indicating behavioral changes."""
        print("Creating behavioral change features...")
        
        # Engagement vs spending mismatch
        # High engagement + low spending = potential convert target
        # Low engagement + high spending = churn risk (not getting value)
        df['engagement_spending_ratio'] = df['engagement_score'] / (df['total_spending_usd'] + 1)
        
        # Activity consistency (are they regular or sporadic?)
        expected_sessions = df['session_frequency_per_week'] * (df['days_since_registration'] / 7)
        df['activity_consistency'] = df['total_playtime_hours'] / (expected_sessions * df['avg_session_duration_min'] / 60 + 1)
        
        # Early vs late game focus
        # High level but low achievements = rushed progression (may burn out)
        df['progression_balance'] = df['achievement_count'] / (df['level_reached'] + 1)
        
        return df
    
    def create_risk_indicators(self, df):
        """Create explicit churn risk indicators."""
        print("Creating risk indicators...")
        
        # Multiple risk flags
        df['risk_inactive'] = (df['days_since_last_login'] > 7).astype(int)
        df['risk_low_engagement'] = (df['engagement_score'] < 30).astype(int)
        df['risk_declining'] = (df['session_frequency_per_week'] < 2).astype(int)
        df['risk_no_friends'] = (df['friend_count'] < 3).astype(int)
        df['risk_no_spending'] = (df['total_spending_usd'] == 0).astype(int)
        df['risk_short_sessions'] = (df['avg_session_duration_min'] < 15).astype(int)
        
        # Total risk score (count of risk flags)
        risk_cols = [col for col in df.columns if col.startswith('risk_')]
        df['total_risk_flags'] = df[risk_cols].sum(axis=1)
        
        # High risk indicator (3+ risk flags)
        df['high_risk'] = (df['total_risk_flags'] >= 3).astype(int)
        
        return df
    
    def create_segment_features(self, df):
        """Create segment-based features."""
        print("Creating segment features...")
        
        # Segment encoded
        df['segment_encoded'] = self.label_encoder.fit_transform(df['segment'])
        
        # Segment performance indicators
        # Is player performing better/worse than segment average?
        segment_avg = df.groupby('segment')['engagement_score'].transform('mean')
        df['above_segment_avg'] = (df['engagement_score'] > segment_avg).astype(int)
        
        # Segment spending deviation
        segment_spending_avg = df.groupby('segment')['total_spending_usd'].transform('mean')
        df['spending_vs_segment'] = df['total_spending_usd'] - segment_spending_avg
        
        return df
    
    def engineer_features(self, df):
        """Run complete feature engineering pipeline."""
        print("=" * 70)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        print()
        
        df = df.copy()
        
        # Create all feature groups
        df = self.create_engagement_features(df)
        df = self.create_spending_features(df)
        df = self.create_social_features(df)
        df = self.create_progression_features(df)
        df = self.create_time_features(df)
        df = self.create_behavioral_change_features(df)
        df = self.create_risk_indicators(df)
        df = self.create_segment_features(df)
        
        # Store feature names (exclude ID, dates, targets)
        exclude_cols = [
            'player_id', 'registration_date', 'churned_7day', 'churned_30day',
            'churn_probability', 'segment', 'spending_tier', 'lifecycle_stage',
            'last_login_bucket'
        ]
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        print()
        print(f"✓ Created {len(self.feature_names)} features")
        print(f"✓ Original features: {len(df.columns) - len(self.feature_names)}")
        print(f"✓ Total columns: {len(df.columns)}")
        
        return df
    
    def prepare_model_features(self, df, target_col='churned_7day'):
        """Prepare features for modeling."""
        print()
        print("=" * 70)
        print("PREPARING MODEL FEATURES")
        print("=" * 70)
        print()
        
        # Select only numeric features for modeling
        numeric_features = df[self.feature_names].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_features]
        y = df[target_col]
        
        # Handle any remaining NaN/inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Target distribution:")
        print(y.value_counts(normalize=True))
        print(f"✓ Churn rate: {y.mean():.2%}")
        
        return X, y, numeric_features
    
    def get_feature_importance_groups(self):
        """Return feature groups for analysis."""
        groups = {
            'Engagement': [f for f in self.feature_names if 'engagement' in f or 'session' in f or 'playtime' in f],
            'Spending': [f for f in self.feature_names if 'spending' in f or 'purchase' in f or 'transaction' in f],
            'Social': [f for f in self.feature_names if 'social' in f or 'friend' in f or 'chat' in f],
            'Progression': [f for f in self.feature_names if 'level' in f or 'achievement' in f or 'progression' in f],
            'Risk': [f for f in self.feature_names if 'risk' in f or 'inactivity' in f],
            'Time': [f for f in self.feature_names if 'days' in f or 'login' in f or 'recency' in f or 'decay' in f]
        }
        return groups


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("GAMING CHURN PREDICTION - FEATURE ENGINEERING")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('gaming_churn_train.csv')
    test_df = pd.read_csv('gaming_churn_test.csv')
    print(f"✓ Training data: {len(train_df)} records")
    print(f"✓ Test data: {len(test_df)} records")
    
    # Initialize feature engineering
    feature_engineer = ChurnFeatureEngineering()
    
    # Engineer features
    print()
    train_df_engineered = feature_engineer.engineer_features(train_df)
    test_df_engineered = feature_engineer.engineer_features(test_df)
    
    # Prepare model features
    X_train_7day, y_train_7day, features_7day = feature_engineer.prepare_model_features(
        train_df_engineered, target_col='churned_7day'
    )
    X_test_7day, y_test_7day, _ = feature_engineer.prepare_model_features(
        test_df_engineered, target_col='churned_7day'
    )
    
    X_train_30day, y_train_30day, features_30day = feature_engineer.prepare_model_features(
        train_df_engineered, target_col='churned_30day'
    )
    X_test_30day, y_test_30day, _ = feature_engineer.prepare_model_features(
        test_df_engineered, target_col='churned_30day'
    )
    
    # Save engineered datasets
    print()
    print("=" * 70)
    print("SAVING ENGINEERED FEATURES")
    print("=" * 70)
    print()
    
    train_df_engineered.to_csv('gaming_churn_train_engineered.csv', index=False)
    test_df_engineered.to_csv('gaming_churn_test_engineered.csv', index=False)
    print("✓ Saved engineered training data")
    print("✓ Saved engineered test data")
    
    # Save feature lists
    import json
    feature_metadata = {
        'total_features': len(features_7day),
        'feature_names': features_7day,
        'feature_groups': feature_engineer.get_feature_importance_groups(),
        'target_7day_churn_rate': float(y_train_7day.mean()),
        'target_30day_churn_rate': float(y_train_30day.mean())
    }
    
    with open('feature_metadata.json', 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print("✓ Saved feature metadata")
    
    # Display feature groups
    print()
    print("=" * 70)
    print("FEATURE GROUPS SUMMARY")
    print("=" * 70)
    print()
    groups = feature_engineer.get_feature_importance_groups()
    for group_name, group_features in groups.items():
        print(f"{group_name}: {len(group_features)} features")
        print(f"  Examples: {', '.join(group_features[:3])}")
        print()
    
    print("=" * 70)
    print("✓ FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Train XGBoost model with engineered features")
    print("2. Evaluate feature importance")
    print("3. Deploy model to production")