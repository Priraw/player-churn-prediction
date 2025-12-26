"""
Gaming Churn Prediction - Synthetic Data Generator
====================================================
This script generates realistic gaming behavior data for training churn prediction models.
Based on real gaming industry patterns and player psychology.

Author: Priyanka Rawat
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)

class GamingDataGenerator:
    """
    Generates synthetic gaming player data with realistic behavioral patterns.
    
    Player Segments:
    - Casual: Low engagement, minimal spending
    - Regular: Moderate engagement, occasional spending
    - Hardcore: High engagement, regular spending
    - Whale: Very high spending, moderate-high engagement
    """
    
    def __init__(self, n_players=10000):
        self.n_players = n_players
        self.segment_distribution = {
            'Casual': 0.40,      # 40% of players
            'Regular': 0.35,     # 35% of players
            'Hardcore': 0.20,    # 20% of players
            'Whale': 0.05        # 5% of players (high-value)
        }
        
    def generate_player_segment(self):
        """Assign players to segments based on industry distribution."""
        segments = list(self.segment_distribution.keys())
        probabilities = list(self.segment_distribution.values())
        return np.random.choice(segments, size=self.n_players, p=probabilities)
    
    def generate_registration_dates(self):
        """Generate registration dates (players joined over past 365 days)."""
        days_ago = np.random.exponential(90, self.n_players)  # More recent players
        days_ago = np.clip(days_ago, 1, 365)
        registration_dates = [
            datetime.now() - timedelta(days=int(d)) for d in days_ago
        ]
        return registration_dates, days_ago
    
    def generate_segment_behavior(self, segment):
        """
        Generate behavioral metrics based on player segment.
        
        Returns dictionary with player metrics.
        """
        if segment == 'Casual':
            # Casual players: low engagement, rare spending
            total_playtime_hours = max(1, np.random.exponential(15))
            session_frequency = max(0.5, np.random.gamma(2, 1))  # ~2 sessions/week
            avg_session_duration = max(5, np.random.gamma(3, 5))  # ~15 min sessions
            total_spending = np.random.choice([0, 0, 0, 5, 10], p=[0.7, 0.15, 0.1, 0.03, 0.02])
            achievement_rate = 0.3  # Complete 30% of achievements
            friend_count = max(0, int(np.random.gamma(2, 2)))
            chat_activity = max(0, int(np.random.exponential(10)))
            
        elif segment == 'Regular':
            # Regular players: moderate engagement, occasional purchases
            total_playtime_hours = max(5, np.random.gamma(8, 5))
            session_frequency = max(1, np.random.gamma(5, 1))  # ~5 sessions/week
            avg_session_duration = max(10, np.random.gamma(5, 8))  # ~40 min sessions
            total_spending = max(0, np.random.gamma(3, 10))  # $0-50 typical
            achievement_rate = 0.5  # Complete 50% of achievements
            friend_count = max(1, int(np.random.gamma(5, 3)))
            chat_activity = max(5, int(np.random.gamma(5, 10)))
            
        elif segment == 'Hardcore':
            # Hardcore players: high engagement, regular spending
            total_playtime_hours = max(20, np.random.gamma(15, 10))
            session_frequency = max(3, np.random.gamma(8, 1.5))  # ~12 sessions/week
            avg_session_duration = max(30, np.random.gamma(10, 10))  # ~100 min sessions
            total_spending = max(10, np.random.gamma(5, 20))  # $50-150 typical
            achievement_rate = 0.75  # Complete 75% of achievements
            friend_count = max(5, int(np.random.gamma(8, 5)))
            chat_activity = max(20, int(np.random.gamma(10, 15)))
            
        else:  # Whale
            # Whale players: very high spending, strong engagement
            total_playtime_hours = max(30, np.random.gamma(12, 15))
            session_frequency = max(4, np.random.gamma(7, 1.5))  # ~10 sessions/week
            avg_session_duration = max(20, np.random.gamma(8, 12))  # ~96 min sessions
            total_spending = max(100, np.random.gamma(8, 100))  # $500+ typical
            achievement_rate = 0.65  # Complete 65% of achievements
            friend_count = max(10, int(np.random.gamma(10, 6)))
            chat_activity = max(50, int(np.random.gamma(15, 20)))
        
        return {
            'total_playtime_hours': round(total_playtime_hours, 1),
            'session_frequency': round(session_frequency, 1),
            'avg_session_duration': round(avg_session_duration, 1),
            'total_spending': round(total_spending, 2),
            'achievement_rate': achievement_rate,
            'friend_count': friend_count,
            'chat_activity': chat_activity
        }
    
    def calculate_churn_probability(self, player_data, days_since_reg):
        """
        Calculate churn probability based on behavioral signals.
        
        Key churn indicators:
        - Days since last login (strongest predictor)
        - Declining session frequency
        - Low engagement metrics
        - Spending patterns
        """
        # Base churn probability by segment
        segment_base_churn = {
            'Casual': 0.35,
            'Regular': 0.20,
            'Hardcore': 0.10,
            'Whale': 0.08
        }
        
        base_churn = segment_base_churn[player_data['segment']]
        
        # Engagement score (0-1, higher = more engaged)
        engagement = min(1.0, (
            player_data['session_frequency'] / 14 * 0.3 +
            player_data['avg_session_duration'] / 120 * 0.2 +
            player_data['total_playtime_hours'] / 200 * 0.2 +
            player_data['achievement_rate'] * 0.15 +
            min(1, player_data['friend_count'] / 20) * 0.15
        ))
        
        # Days since last login (key churn predictor)
        # Use power law: recent activity = much lower churn
        recency_factor = np.random.exponential(3)  # Average 3 days
        if engagement < 0.3:  # Low engagement = longer gaps
            recency_factor *= 2
        days_since_last_login = max(0, int(recency_factor))
        
        # Churn probability increases exponentially with inactivity
        inactivity_penalty = min(0.5, days_since_last_login / 14 * 0.5)
        
        # Spending protection (high spenders less likely to churn)
        spending_protection = min(0.2, player_data['total_spending'] / 500 * 0.2)
        
        # Time-based churn (new players churn more in first 30 days)
        if days_since_reg < 30:
            new_player_risk = 0.15
        else:
            new_player_risk = 0
        
        # Final churn probability
        churn_prob = base_churn + inactivity_penalty + new_player_risk - (engagement * 0.3) - spending_protection
        churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05), 0, 0.95)
        
        return churn_prob, days_since_last_login
    
    def generate_dataset(self):
        """Generate complete dataset with all features."""
        print(f"Generating dataset for {self.n_players} players...")
        
        # Generate segments
        segments = self.generate_player_segment()
        
        # Generate registration dates
        registration_dates, days_since_registration = self.generate_registration_dates()
        
        # Generate player data
        players = []
        for i in range(self.n_players):
            segment = segments[i]
            days_since_reg = days_since_registration[i]
            
            # Generate behavioral data
            behavior = self.generate_segment_behavior(segment)
            
            # Calculate achievement count
            max_achievements = 150  # Total achievements in game
            achievement_count = int(behavior['achievement_rate'] * max_achievements)
            
            # Calculate level (based on playtime)
            level = min(100, int(behavior['total_playtime_hours'] / 2))
            
            # Number of in-game purchases
            if behavior['total_spending'] == 0:
                purchase_count = 0
            else:
                purchase_count = max(1, int(behavior['total_spending'] / 15))
            
            # Calculate churn
            player_data = {
                'segment': segment,
                'session_frequency': behavior['session_frequency'],
                'avg_session_duration': behavior['avg_session_duration'],
                'total_playtime_hours': behavior['total_playtime_hours'],
                'achievement_rate': behavior['achievement_rate'],
                'friend_count': behavior['friend_count'],
                'total_spending': behavior['total_spending']
            }
            
            churn_prob, days_since_last_login = self.calculate_churn_probability(
                player_data, days_since_reg
            )
            
            # Determine actual churn status
            churned_7day = 1 if days_since_last_login >= 7 else 0
            churned_30day = 1 if days_since_last_login >= 30 else 0
            
            # Compile player record
            player = {
                'player_id': f'P{1000 + i}',
                'registration_date': registration_dates[i].strftime('%Y-%m-%d'),
                'days_since_registration': int(days_since_reg),
                'segment': segment,
                'total_playtime_hours': behavior['total_playtime_hours'],
                'session_frequency_per_week': behavior['session_frequency'],
                'avg_session_duration_min': behavior['avg_session_duration'],
                'total_spending_usd': behavior['total_spending'],
                'days_since_last_login': days_since_last_login,
                'achievement_count': achievement_count,
                'friend_count': behavior['friend_count'],
                'chat_messages_sent': behavior['chat_activity'],
                'in_game_purchases_count': purchase_count,
                'level_reached': level,
                'churn_probability': round(churn_prob, 4),
                'churned_7day': churned_7day,
                'churned_30day': churned_30day
            }
            
            players.append(player)
        
        df = pd.DataFrame(players)
        
        print(f"✓ Generated {len(df)} player records")
        print(f"✓ Segment distribution:")
        print(df['segment'].value_counts(normalize=True))
        print(f"✓ 7-day churn rate: {df['churned_7day'].mean():.2%}")
        print(f"✓ 30-day churn rate: {df['churned_30day'].mean():.2%}")
        
        return df
    
    def save_dataset(self, df, filepath='gaming_churn_data.csv'):
        """Save dataset to CSV."""
        df.to_csv(filepath, index=False)
        print(f"✓ Dataset saved to {filepath}")
        
        # Save data dictionary
        data_dict = {
            'dataset_info': {
                'name': 'Gaming Player Churn Dataset',
                'rows': len(df),
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'author': 'Priyanka Rawat'
            },
            'features': {
                'player_id': 'Unique player identifier',
                'registration_date': 'Date player joined game',
                'days_since_registration': 'Days since player registered',
                'segment': 'Player segment (Casual/Regular/Hardcore/Whale)',
                'total_playtime_hours': 'Total hours played since registration',
                'session_frequency_per_week': 'Average sessions per week',
                'avg_session_duration_min': 'Average session length in minutes',
                'total_spending_usd': 'Total amount spent in USD',
                'days_since_last_login': 'Days since last login (KEY PREDICTOR)',
                'achievement_count': 'Number of achievements unlocked',
                'friend_count': 'Number of in-game friends',
                'chat_messages_sent': 'Total chat messages sent',
                'in_game_purchases_count': 'Number of purchases made',
                'level_reached': 'Current player level (1-100)',
                'churn_probability': 'Calculated churn probability',
                'churned_7day': 'Target: 1 if no login in 7+ days',
                'churned_30day': 'Target: 1 if no login in 30+ days'
            },
            'segment_definitions': {
                'Casual': 'Low engagement, minimal spending, ~40% of players',
                'Regular': 'Moderate engagement, occasional spending, ~35% of players',
                'Hardcore': 'High engagement, regular spending, ~20% of players',
                'Whale': 'Very high spending, moderate-high engagement, ~5% of players'
            }
        }
        
        dict_filepath = filepath.replace('.csv', '_dictionary.json')
        with open(dict_filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)
        print(f"✓ Data dictionary saved to {dict_filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GAMING CHURN PREDICTION - DATA GENERATION")
    print("=" * 70)
    print()
    
    # Generate dataset
    generator = GamingDataGenerator(n_players=10000)
    df = generator.generate_dataset()
    
    print()
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print()
    print(df.describe())
    print()
    print("Correlations with 7-day churn:")
    print(df.corr()['churned_7day'].sort_values(ascending=False))
    
    # Save dataset
    print()
    print("=" * 70)
    print("SAVING FILES")
    print("=" * 70)
    print()
    generator.save_dataset(df, 'gaming_churn_data.csv')
    
    # Create train/test split files
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['segment'])
    train_df.to_csv('gaming_churn_train.csv', index=False)
    test_df.to_csv('gaming_churn_test.csv', index=False)
    
    print(f"✓ Training set saved: {len(train_df)} records (gaming_churn_train.csv)")
    print(f"✓ Test set saved: {len(test_df)} records (gaming_churn_test.csv)")
    
    print()
    print("=" * 70)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run EDA notebook to explore the data")
    print("2. Train churn prediction model")
    print("3. Deploy FastAPI backend")