"""FastAPI application for fraud detection service."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

from ..models.predict import FraudPredictor
from ..utils.config import Config, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Production-ready API for detecting fraudulent insurance claims",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[FraudPredictor] = None
config: Optional[Config] = None


# Pydantic models for request/response
class ClaimFeatures(BaseModel):
    """Input features for a single claim."""
    
    # Driver information
    age_of_driver: int = Field(..., ge=18, le=100, description="Age of the driver")
    gender: str = Field(..., description="Gender (M/F)")
    marital_status: int = Field(..., ge=0, le=1, description="Marital status (0=Single, 1=Married)")
    safety_rating: int = Field(..., ge=0, le=100, description="Safety rating")
    annual_income: float = Field(..., gt=0, description="Annual income")
    high_education: int = Field(..., ge=0, le=1, description="Higher education (0/1)")
    
    # Address and location
    address_change: int = Field(..., ge=0, le=1, description="Recent address change (0/1)")
    property_status: str = Field(..., description="Property status (Own/Rent)")
    zip_code: int = Field(..., description="ZIP code")
    
    # Claim information
    claim_date: str = Field(..., description="Claim date (MM/DD/YYYY)")
    claim_day_of_week: str = Field(..., description="Day of week")
    accident_site: str = Field(..., description="Accident location")
    past_num_of_claims: int = Field(..., ge=0, description="Number of past claims")
    witness_present: int = Field(..., ge=0, le=1, description="Witness present (0/1)")
    liab_prct: int = Field(..., ge=0, le=100, description="Liability percentage")
    channel: str = Field(..., description="Channel")
    police_report: int = Field(..., ge=0, le=1, description="Police report filed (0/1)")
    
    # Vehicle information
    age_of_vehicle: int = Field(..., ge=0, description="Age of vehicle in years")
    vehicle_category: str = Field(..., description="Vehicle category")
    vehicle_price: float = Field(..., gt=0, description="Vehicle price")
    vehicle_color: str = Field(..., description="Vehicle color")
    
    # Financial information
    total_claim: float = Field(..., gt=0, description="Total claim amount")
    injury_claim: float = Field(..., ge=0, description="Injury claim amount")
    policy_deductible: int = Field(..., gt=0, description="Policy deductible")
    annual_premium: float = Field(..., gt=0, description="Annual premium")
    
    # Additional information
    days_open: float = Field(..., gt=0, description="Days case has been open")
    form_defects: int = Field(..., ge=0, description="Number of form defects")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['M', 'F']:
            raise ValueError('Gender must be M or F')
        return v
    
    @validator('property_status')
    def validate_property_status(cls, v):
        if v not in ['Own', 'Rent']:
            raise ValueError('Property status must be Own or Rent')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "age_of_driver": 39,
                "gender": "M",
                "marital_status": 1,
                "safety_rating": 73,
                "annual_income": 58612.8,
                "high_education": 1,
                "address_change": 0,
                "property_status": "Own",
                "zip_code": 50048,
                "claim_date": "8/12/2023",
                "claim_day_of_week": "Saturday",
                "accident_site": "Highway",
                "past_num_of_claims": 0,
                "witness_present": 0,
                "liab_prct": 25,
                "channel": "Phone",
                "police_report": 0,
                "age_of_vehicle": 8,
                "vehicle_category": "Large",
                "vehicle_price": 24360.59,
                "vehicle_color": "silver",
                "total_claim": 26633.27,
                "injury_claim": 5196.55,
                "policy_deductible": 1000,
                "annual_premium": 1406.91,
                "days_open": 8.64,
                "form_defects": 5
            }
        }


class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    
    is_fraud: bool = Field(..., description="Whether claim is fraudulent")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    risk_level: str = Field(..., description="Risk level (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.23,
                "confidence": 0.54,
                "risk_level": "LOW",
                "timestamp": "2024-01-27T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    claims: List[ClaimFeatures] = Field(..., description="List of claims to predict")
    
    class Config:
        schema_extra = {
            "example": {
                "claims": [
                    {
                        "age_of_driver": 39,
                        "gender": "M",
                        "marital_status": 1,
                        "safety_rating": 73,
                        "annual_income": 58612.8,
                        # ... (abbreviated for brevity)
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global predictor, config
    
    try:
        config = Config()
        
        # Try to load the best model
        models_dir = config.models_dir
        model_files = list(models_dir.glob("*_model.pkl"))
        
        if model_files:
            # Load the most recent model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            predictor = FraudPredictor(model_path=str(latest_model), config=config)
            logger.info(f"Model loaded successfully: {latest_model}")
        else:
            logger.warning("No trained model found. Train a model first.")
            predictor = None
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        predictor = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Insurance Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(claim: ClaimFeatures):
    """
    Predict fraud for a single insurance claim.
    
    Args:
        claim: Claim features
        
    Returns:
        Fraud prediction with probability and risk level
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        # Convert to dict
        claim_dict = claim.dict()
        
        # Make prediction
        result = predictor.predict_single(claim_dict)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_fraud_batch(request: BatchPredictionRequest):
    """
    Predict fraud for multiple insurance claims.
    
    Args:
        request: Batch of claims
        
    Returns:
        List of predictions
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        # Convert to DataFrame
        claims_data = [claim.dict() for claim in request.claims]
        claims_df = pd.DataFrame(claims_data)
        
        # Make predictions
        results_df = predictor.predict_batch(claims_df)
        
        # Format response
        predictions = []
        for _, row in results_df.iterrows():
            predictions.append({
                "is_fraud": bool(row['is_fraud']),
                "fraud_probability": float(row['fraud_probability']),
                "risk_level": row['risk_level'],
                "timestamp": datetime.now().isoformat()
            })
        
        return {"predictions": predictions, "count": len(predictions)}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return predictor.get_model_info()


@app.post("/model/threshold", tags=["Model"])
async def set_prediction_threshold(threshold: float):
    """
    Set custom prediction threshold.
    
    Args:
        threshold: Threshold value between 0 and 1
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        predictor.set_threshold(threshold)
        return {"message": f"Threshold set to {threshold}"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )