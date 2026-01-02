"""
Player Churn Prediction - FastAPI Backend
==========================================
Production-ready API for serving churn predictions.

Features:
- Real-time predictions
- Batch predictions
- Feature importance endpoint
- Model health checks
- Automatic input validation

Author: Priyanka Rawat
Date: December 2025

Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Player Churn Prediction API",
    description="AI-powered player retention analytics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    model_7day = joblib.load('churn_model_7day.pkl')
    model_30day = joblib.load('churn_model_30day.pkl')
    
    with open('feature_metadata.json', 'r') as f:
        import json
        feature_metadata = json.load(f)
        FEATURE_NAMES = feature_metadata['feature_names']
    
    logger.info("✓ Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    model_7day = None
    model_30day = None
    FEATURE_NAMES = []

# Pydantic models
class PlayerData(BaseModel):
    player_id: Optional[str] = Field(None, description="Player identifier")
    segment: str = Field(..., description="Player segment: Casual, Regular, Hardcore, Whale")
    
    total_playtime_hours: float = Field(..., ge=0)
    session_frequency_per_week: float = Field(..., ge=0, le=50)
    avg_session_duration_min: float = Field(..., ge=0)
    total_spending_usd: float = Field(..., ge=0)
    in_game_purchases_count: int = Field(..., ge=0)
    days_since_registration: int = Field(..., ge=1)
    days_since_last_login: int = Field(..., ge=0)
    achievement_count: int = Field(..., ge=0, le=150)
    level_reached: int = Field(..., ge=1, le=100)
    friend_count: int = Field(..., ge=0)
    chat_messages_sent: int = Field(..., ge=0)
    
    @field_validator('segment')
    @classmethod
    def validate_segment(cls, v):
        allowed = ['Casual', 'Regular', 'Hardcore', 'Whale']
        if v not in allowed:
            raise ValueError(f'Segment must be one of {allowed}')
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "player_id": "P1234",
                "segment": "Regular",
                "total_playtime_hours": 45.5,
                "session_frequency_per_week": 5.2,
                "avg_session_duration_min": 38.5,
                "total_spending_usd": 29.99,
                "in_game_purchases_count": 3,
                "days_since_registration": 60,
                "days_since_last_login": 2,
                "achievement_count": 42,
                "level_reached": 28,
                "friend_count": 12,
                "chat_messages_sent": 156
            }
        }
    }


class BatchPlayerData(BaseModel):
    players: List[PlayerData] = Field(..., max_length=1000, description="List of players (max 1000)")


class PredictionResponse(BaseModel):
    player_id: Optional[str]
    churn_prob_7day: float
    churn_prob_30day: float
    risk_level: str
    recommended_actions: List[str]
    prediction_timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]  # Fixed: changed 'any' to 'Any'


class FeatureImportanceResponse(BaseModel):
    model: str
    features: List[Dict[str, float]]


class HealthResponse(BaseModel):
    status: str
    model_7day_loaded: bool
    model_30day_loaded: bool
    timestamp: str


# Feature engineering function
def engineer_features(player_data: Dict) -> Dict:
    """Engineer features from raw player data."""
    
    engagement_score = (
        (player_data['session_frequency_per_week'] / 14) * 25 +
        (player_data['avg_session_duration_min'] / 120) * 25 +
        (player_data['total_playtime_hours'] / 200) * 25 +
        (player_data['achievement_count'] / 150) * 25
    )
    engagement_score = min(100, engagement_score)
    
    play_intensity = player_data['total_playtime_hours'] / (player_data['days_since_registration'] + 1)
    session_consistency = player_data['session_frequency_per_week'] / 7
    achievement_velocity = player_data['achievement_count'] / (player_data['total_playtime_hours'] + 1)
    recency_score = np.exp(-player_data['days_since_last_login'] / 7)
    
    spending_per_hour = player_data['total_spending_usd'] / (player_data['total_playtime_hours'] + 1)
    total_sessions = player_data['session_frequency_per_week'] * (player_data['days_since_registration'] / 7)
    spending_per_session = player_data['total_spending_usd'] / (total_sessions + 1)
    purchase_frequency = player_data['in_game_purchases_count'] / (player_data['days_since_registration'] / 7 + 1)
    avg_transaction_value = player_data['total_spending_usd'] / (player_data['in_game_purchases_count'] + 1)
    is_spender = 1 if player_data['total_spending_usd'] > 0 else 0
    
    social_score = (
        np.log1p(player_data['friend_count']) * 50 +
        np.log1p(player_data['chat_messages_sent']) * 50
    )
    chat_per_friend = player_data['chat_messages_sent'] / (player_data['friend_count'] + 1)
    social_ratio = player_data['friend_count'] / (player_data['total_playtime_hours'] + 1)
    is_social = 1 if (player_data['friend_count'] > 5 or player_data['chat_messages_sent'] > 50) else 0
    
    leveling_speed = player_data['level_reached'] / (player_data['total_playtime_hours'] + 1)
    achievement_completion_rate = player_data['achievement_count'] / 150
    progression_score = (player_data['level_reached'] / 100) * 60 + achievement_completion_rate * 40
    
    inactivity_risk = 1 - np.exp(-player_data['days_since_last_login'] / 7)
    activity_decay = player_data['days_since_last_login'] / (player_data['days_since_registration'] + 1)
    expected_next_login = 7 / (player_data['session_frequency_per_week'] + 0.1)
    overdue_for_login = 1 if player_data['days_since_last_login'] > expected_next_login else 0
    
    engagement_spending_ratio = engagement_score / (player_data['total_spending_usd'] + 1)
    progression_balance = player_data['achievement_count'] / (player_data['level_reached'] + 1)
    
    risk_inactive = 1 if player_data['days_since_last_login'] > 7 else 0
    risk_low_engagement = 1 if engagement_score < 30 else 0
    risk_declining = 1 if player_data['session_frequency_per_week'] < 2 else 0
    risk_no_friends = 1 if player_data['friend_count'] < 3 else 0
    risk_no_spending = 1 if player_data['total_spending_usd'] == 0 else 0
    risk_short_sessions = 1 if player_data['avg_session_duration_min'] < 15 else 0
    total_risk_flags = (risk_inactive + risk_low_engagement + risk_declining + 
                       risk_no_friends + risk_no_spending + risk_short_sessions)
    high_risk = 1 if total_risk_flags >= 3 else 0
    
    segment_mapping = {'Casual': 0, 'Regular': 1, 'Hardcore': 2, 'Whale': 3}
    segment_encoded = segment_mapping.get(player_data['segment'], 0)
    
    features = {
        'days_since_registration': player_data['days_since_registration'],
        'total_playtime_hours': player_data['total_playtime_hours'],
        'session_frequency_per_week': player_data['session_frequency_per_week'],
        'avg_session_duration_min': player_data['avg_session_duration_min'],
        'total_spending_usd': player_data['total_spending_usd'],
        'days_since_last_login': player_data['days_since_last_login'],
        'achievement_count': player_data['achievement_count'],
        'friend_count': player_data['friend_count'],
        'chat_messages_sent': player_data['chat_messages_sent'],
        'in_game_purchases_count': player_data['in_game_purchases_count'],
        'level_reached': player_data['level_reached'],
        'engagement_score': engagement_score,
        'play_intensity': play_intensity,
        'session_consistency': session_consistency,
        'achievement_velocity': achievement_velocity,
        'recency_score': recency_score,
        'spending_per_hour': spending_per_hour,
        'spending_per_session': spending_per_session,
        'purchase_frequency': purchase_frequency,
        'avg_transaction_value': avg_transaction_value,
        'is_spender': is_spender,
        'social_score': social_score,
        'chat_per_friend': chat_per_friend,
        'social_ratio': social_ratio,
        'is_social': is_social,
        'leveling_speed': leveling_speed,
        'achievement_completion_rate': achievement_completion_rate,
        'progression_score': progression_score,
        'inactivity_risk': inactivity_risk,
        'activity_decay': activity_decay,
        'overdue_for_login': overdue_for_login,
        'engagement_spending_ratio': engagement_spending_ratio,
        'progression_balance': progression_balance,
        'risk_inactive': risk_inactive,
        'risk_low_engagement': risk_low_engagement,
        'risk_declining': risk_declining,
        'risk_no_friends': risk_no_friends,
        'risk_no_spending': risk_no_spending,
        'risk_short_sessions': risk_short_sessions,
        'total_risk_flags': total_risk_flags,
        'high_risk': high_risk,
        'segment_encoded': segment_encoded
    }
    
    return features


def get_risk_level(churn_prob_7day: float) -> str:
    if churn_prob_7day >= 0.8:
        return "Critical"
    elif churn_prob_7day >= 0.6:
        return "High"
    elif churn_prob_7day >= 0.3:
        return "Medium"
    else:
        return "Low"


def get_recommended_actions(player_data: Dict, churn_prob_7day: float) -> List[str]:
    actions = []
    
    if player_data['days_since_last_login'] > 7:
        actions.append("🚨 URGENT: Win-back email with 50% off premium currency")
    elif player_data['days_since_last_login'] > 3:
        actions.append("📱 Push notification: Daily reward waiting")
    
    if player_data['session_frequency_per_week'] < 3:
        actions.append("🎯 Enable daily quest streak")
    
    if player_data['total_spending_usd'] > 50 and churn_prob_7day > 0.5:
        actions.append("⭐ VIP: Assign personal account manager")
    
    if player_data['friend_count'] < 3:
        actions.append("👥 Social boost: Friend referral bonus")
    
    if len(actions) == 0:
        actions.append("✅ Player healthy: Continue regular engagement")
    
    return actions


# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Player Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if (model_7day and model_30day) else "degraded",
        model_7day_loaded=model_7day is not None,
        model_30day_loaded=model_30day is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(player: PlayerData):
    if not model_7day or not model_30day:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        player_dict = player.model_dump()
        features = engineer_features(player_dict)
        feature_vector = pd.DataFrame([features])[FEATURE_NAMES]
        feature_vector = feature_vector.fillna(0).replace([np.inf, -np.inf], 0)
        
        churn_prob_7day = float(model_7day.predict_proba(feature_vector)[0, 1])
        churn_prob_30day = float(model_30day.predict_proba(feature_vector)[0, 1])
        
        risk_level = get_risk_level(churn_prob_7day)
        recommended_actions = get_recommended_actions(player_dict, churn_prob_7day)
        
        return PredictionResponse(
            player_id=player.player_id,
            churn_prob_7day=round(churn_prob_7day, 4),
            churn_prob_30day=round(churn_prob_30day, 4),
            risk_level=risk_level,
            recommended_actions=recommended_actions,
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance/{model_type}", response_model=FeatureImportanceResponse)
async def get_feature_importance(model_type: str):
    if model_type not in ['7day', '30day']:
        raise HTTPException(status_code=400, detail="model_type must be '7day' or '30day'")
    
    model = model_7day if model_type == '7day' else model_30day
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        importance = model.feature_importances_
        features = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(FEATURE_NAMES, importance)
        ]
        features = sorted(features, key=lambda x: x['importance'], reverse=True)
        
        return FeatureImportanceResponse(
            model=model_type,
            features=features[:30]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)