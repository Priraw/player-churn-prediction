"""
Gaming Churn Prediction - FastAPI Backend
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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZE APP
# ============================================================================

app = FastAPI(
    title="Gaming Churn Prediction API",
    description="AI-powered player retention analytics",
    version="1.0.0",
    contact={
        "name": "Priyanka Rawat",
        "url": "https://www.priyanka-rawat.com/",
        "email": "pri00raw@gmail.com"
    }
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODELS
# ============================================================================

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

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PlayerData(BaseModel):
    """Input schema for single player prediction."""
    
    player_id: Optional[str] = Field(None, description="Player identifier")
    segment: str = Field(..., description="Player segment: Casual, Regular, Hardcore, Whale")
    
    # Core engagement metrics
    total_playtime_hours: float = Field(..., ge=0, description="Total hours played")
    session_frequency_per_week: float = Field(..., ge=0, le=50, description="Sessions per week")
    avg_session_duration_min: float = Field(..., ge=0, description="Average session duration in minutes")
    
    # Spending metrics
    total_spending_usd: float = Field(..., ge=0, description="Total USD spent")
    in_game_purchases_count: int = Field(..., ge=0, description="Number of purchases")
    
    # Activity metrics
    days_since_registration: int = Field(..., ge=1, description="Days since registration")
    days_since_last_login: int = Field(..., ge=0, description="Days since last login")
    
    # Progression metrics
    achievement_count: int = Field(..., ge=0, le=150, description="Achievements unlocked (max 150)")
    level_reached: int = Field(..., ge=1, le=100, description="Current level (1-100)")
    
    # Social metrics
    friend_count: int = Field(..., ge=0, description="Number of friends")
    chat_messages_sent: int = Field(..., ge=0, description="Total chat messages")
    
    @validator('segment')
    def validate_segment(cls, v):
        allowed = ['Casual', 'Regular', 'Hardcore', 'Whale']
        if v not in allowed:
            raise ValueError(f'Segment must be one of {allowed}')
        return v
    
    class Config:
        schema_extra = {
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


class BatchPlayerData(BaseModel):
    """Input schema for batch predictions."""
    players: List[PlayerData] = Field(..., max_items=1000, description="List of players (max 1000)")


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    player_id: Optional[str]
    churn_prob_7day: float = Field(..., description="7-day churn probability (0-1)")
    churn_prob_30day: float = Field(..., description="30-day churn probability (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High, Critical")
    recommended_actions: List[str] = Field(..., description="Recommended retention actions")
    prediction_timestamp: str = Field(..., description="ISO timestamp of prediction")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    summary: Dict[str, any]


class FeatureImportanceResponse(BaseModel):
    """Response schema for feature importance."""
    model: str
    features: List[Dict[str, float]]


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_7day_loaded: bool
    model_30day_loaded: bool
    timestamp: str

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(player_data: Dict) -> Dict:
    """
    Engineer features from raw player data.
    Matches the feature engineering pipeline used in training.
    """
    
    # Engagement features
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
    
    # Spending features
    spending_per_hour = player_data['total_spending_usd'] / (player_data['total_playtime_hours'] + 1)
    total_sessions = player_data['session_frequency_per_week'] * (player_data['days_since_registration'] / 7)
    spending_per_session = player_data['total_spending_usd'] / (total_sessions + 1)
    purchase_frequency = player_data['in_game_purchases_count'] / (player_data['days_since_registration'] / 7 + 1)
    avg_transaction_value = player_data['total_spending_usd'] / (player_data['in_game_purchases_count'] + 1)
    is_spender = 1 if player_data['total_spending_usd'] > 0 else 0
    
    # Social features
    social_score = (
        np.log1p(player_data['friend_count']) * 50 +
        np.log1p(player_data['chat_messages_sent']) * 50
    )
    chat_per_friend = player_data['chat_messages_sent'] / (player_data['friend_count'] + 1)
    social_ratio = player_data['friend_count'] / (player_data['total_playtime_hours'] + 1)
    is_social = 1 if (player_data['friend_count'] > 5 or player_data['chat_messages_sent'] > 50) else 0
    
    # Progression features
    leveling_speed = player_data['level_reached'] / (player_data['total_playtime_hours'] + 1)
    achievement_completion_rate = player_data['achievement_count'] / 150
    progression_score = (player_data['level_reached'] / 100) * 60 + achievement_completion_rate * 40
    
    # Time features
    inactivity_risk = 1 - np.exp(-player_data['days_since_last_login'] / 7)
    activity_decay = player_data['days_since_last_login'] / (player_data['days_since_registration'] + 1)
    expected_next_login = 7 / (player_data['session_frequency_per_week'] + 0.1)
    overdue_for_login = 1 if player_data['days_since_last_login'] > expected_next_login else 0
    
    # Behavioral change features
    engagement_spending_ratio = engagement_score / (player_data['total_spending_usd'] + 1)
    progression_balance = player_data['achievement_count'] / (player_data['level_reached'] + 1)
    
    # Risk indicators
    risk_inactive = 1 if player_data['days_since_last_login'] > 7 else 0
    risk_low_engagement = 1 if engagement_score < 30 else 0
    risk_declining = 1 if player_data['session_frequency_per_week'] < 2 else 0
    risk_no_friends = 1 if player_data['friend_count'] < 3 else 0
    risk_no_spending = 1 if player_data['total_spending_usd'] == 0 else 0
    risk_short_sessions = 1 if player_data['avg_session_duration_min'] < 15 else 0
    total_risk_flags = (risk_inactive + risk_low_engagement + risk_declining + 
                       risk_no_friends + risk_no_spending + risk_short_sessions)
    high_risk = 1 if total_risk_flags >= 3 else 0
    
    # Segment encoding
    segment_mapping = {'Casual': 0, 'Regular': 1, 'Hardcore': 2, 'Whale': 3}
    segment_encoded = segment_mapping.get(player_data['segment'], 0)
    
    # Compile all features
    features = {
        # Original features
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
        
        # Engineered features
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_level(churn_prob_7day: float) -> str:
    """Categorize churn probability into risk levels."""
    if churn_prob_7day >= 0.8:
        return "Critical"
    elif churn_prob_7day >= 0.6:
        return "High"
    elif churn_prob_7day >= 0.3:
        return "Medium"
    else:
        return "Low"


def get_recommended_actions(player_data: Dict, churn_prob_7day: float) -> List[str]:
    """Generate personalized retention recommendations."""
    actions = []
    
    if player_data['days_since_last_login'] > 7:
        actions.append("🚨 URGENT: Win-back email with 50% off premium currency")
    elif player_data['days_since_last_login'] > 3:
        actions.append("📱 Push notification: Daily reward waiting + new content alert")
    
    if player_data['session_frequency_per_week'] < 3:
        actions.append("🎯 Enable daily quest streak with escalating rewards")
    
    if player_data['total_spending_usd'] > 50 and churn_prob_7day > 0.5:
        actions.append("⭐ VIP: Assign personal account manager + exclusive content access")
    
    if player_data['avg_session_duration_min'] < 20:
        actions.append("🎮 Content discovery: Highlight unexplored features")
    
    if player_data['friend_count'] < 3:
        actions.append("👥 Social boost: Friend referral bonus + guild invitation")
    
    if player_data['achievement_count'] < 30:
        actions.append("🏆 Achievement campaign: Double achievement XP weekend")
    
    if len(actions) == 0:
        actions.append("✅ Player healthy: Continue regular engagement content")
    
    return actions

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Gaming Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "author": "Priyanka Rawat"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if (model_7day and model_30day) else "degraded",
        model_7day_loaded=model_7day is not None,
        model_30day_loaded=model_30day is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(player: PlayerData):
    """
    Predict churn probability for a single player.
    
    Returns 7-day and 30-day churn probabilities with risk level and recommendations.
    """
    if not model_7day or not model_30day:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to dict
        player_dict = player.dict()
        
        # Engineer features
        features = engineer_features(player_dict)
        
        # Create feature vector (match training feature order)
        feature_vector = pd.DataFrame([features])[FEATURE_NAMES]
        feature_vector = feature_vector.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Predictions
        churn_prob_7day = float(model_7day.predict_proba(feature_vector)[0, 1])
        churn_prob_30day = float(model_30day.predict_proba(feature_vector)[0, 1])
        
        # Risk level
        risk_level = get_risk_level(churn_prob_7day)
        
        # Recommendations
        recommended_actions = get_recommended_actions(player_dict, churn_prob_7day)
        
        logger.info(f"Prediction for {player.player_id}: 7d={churn_prob_7day:.3f}, 30d={churn_prob_30day:.3f}")
        
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchPlayerData):
    """
    Predict churn for multiple players (max 1000).
    
    Returns predictions for all players plus summary statistics.
    """
    if not model_7day or not model_30day:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        predictions = []
        
        for player in batch.players:
            pred = await predict_churn(player)
            predictions.append(pred)
        
        # Calculate summary statistics
        high_risk = sum(1 for p in predictions if p.risk_level in ["High", "Critical"])
        avg_churn_7day = np.mean([p.churn_prob_7day for p in predictions])
        avg_churn_30day = np.mean([p.churn_prob_30day for p in predictions])
        
        summary = {
            "total_players": len(predictions),
            "high_risk_count": high_risk,
            "high_risk_percentage": round(high_risk / len(predictions) * 100, 2),
            "avg_churn_prob_7day": round(avg_churn_7day, 4),
            "avg_churn_prob_30day": round(avg_churn_30day, 4)
        }
        
        logger.info(f"Batch prediction completed: {len(predictions)} players")
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/feature-importance/{model_type}", response_model=FeatureImportanceResponse)
async def get_feature_importance(model_type: str):
    """
    Get feature importance for 7day or 30day model.
    
    Args:
        model_type: Either '7day' or '30day'
    """
    if model_type not in ['7day', '30day']:
        raise HTTPException(status_code=400, detail="model_type must be '7day' or '30day'")
    
    model = model_7day if model_type == '7day' else model_30day
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get feature importance
        importance = model.feature_importances_
        features = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(FEATURE_NAMES, importance)
        ]
        
        # Sort by importance
        features = sorted(features, key=lambda x: x['importance'], reverse=True)
        
        return FeatureImportanceResponse(
            model=model_type,
            features=features[:30]  # Top 30 features
        )
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)