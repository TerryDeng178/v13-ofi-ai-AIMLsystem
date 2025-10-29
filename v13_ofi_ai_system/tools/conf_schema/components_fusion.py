"""
Fusion组件配置Schema
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class FusionThresholds(BaseModel):
    """Fusion阈值配置"""
    fuse_buy: float = Field(..., description="买入阈值")
    fuse_sell: float = Field(..., description="卖出阈值")
    fuse_strong_buy: float = Field(..., description="强买入阈值")
    fuse_strong_sell: float = Field(..., description="强卖出阈值")
    
    @field_validator('fuse_strong_buy')
    @classmethod
    def validate_strong_buy(cls, v, info):
        fuse_buy = info.data.get('fuse_buy', 0.0)
        if v < fuse_buy:
            raise ValueError(f'fuse_strong_buy ({v}) must be >= fuse_buy ({fuse_buy})')
        return v
    
    @field_validator('fuse_strong_sell')
    @classmethod
    def validate_strong_sell(cls, v, info):
        fuse_sell = info.data.get('fuse_sell', 0.0)
        if v > fuse_sell:  # 负数比较
            raise ValueError(f'fuse_strong_sell ({v}) must be <= fuse_sell ({fuse_sell})')
        return v


class FusionConsistency(BaseModel):
    """Fusion一致性配置"""
    min_consistency: float = Field(..., ge=0.0, le=1.0, description="最小一致性")
    strong_min_consistency: float = Field(..., ge=0.0, le=1.0, description="强信号最小一致性")
    
    @field_validator('strong_min_consistency')
    @classmethod
    def validate_strong_consistency(cls, v, info):
        min_cons = info.data.get('min_consistency', 0.0)
        if v < min_cons:
            raise ValueError(f'strong_min_consistency ({v}) must be >= min_consistency ({min_cons})')
        return v


class FusionWeights(BaseModel):
    """Fusion权重配置"""
    w_ofi: float = Field(..., ge=0.0, le=1.0, description="OFI权重")
    w_cvd: float = Field(..., ge=0.0, le=1.0, description="CVD权重")
    
    @field_validator('w_cvd')
    @classmethod
    def validate_weights_sum(cls, v, info):
        w_ofi = info.data.get('w_ofi', 0.0)
        total = w_ofi + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'weights must sum to 1.0, got {total}')
        return v


class FusionSmoothing(BaseModel):
    """Fusion平滑配置"""
    z_window: int = Field(..., gt=0, description="Z窗口大小")
    winsorize_percentile: int = Field(..., ge=1, le=100, description="Winsorize百分位")
    mad_k: float = Field(..., gt=0.0, description="MAD系数")


class FusionConfig(BaseModel):
    """Fusion完整配置"""
    thresholds: FusionThresholds
    consistency: FusionConsistency
    weights: FusionWeights
    smoothing: FusionSmoothing
    
    class Config:
        json_schema_extra = {
            "example": {
                "thresholds": {
                    "fuse_buy": 1.0,
                    "fuse_sell": -1.0,
                    "fuse_strong_buy": 2.3,
                    "fuse_strong_sell": -2.3
                },
                "consistency": {
                    "min_consistency": 0.20,
                    "strong_min_consistency": 0.65
                },
                "weights": {
                    "w_ofi": 0.6,
                    "w_cvd": 0.4
                },
                "smoothing": {
                    "z_window": 60,
                    "winsorize_percentile": 95,
                    "mad_k": 2.0
                }
            }
        }


