"""
Strategies package for trading strategies
"""

from .base_strategy import BaseStrategy, StrategyConfig
from .mean_reversion_strategy import (PortfolioMeanReversionConfig,
                                      PortfolioMeanReversionStrategy)
from .momentum_strategy import AdaptiveMomentumConfig, MomentumTrendStrategy
from .nifty_shop_strategy import NiftyShopConfig, NiftyShopStrategy
from .pairs_strategy import PairsConfig, PairsStrategy
from .pmv_momentum_strategy import PMVMomentumConfig, PMVMomentumStrategy
from .statistical_trend_strategy import (StatisticalTrendConfig,
                                         StatisticalTrendStrategy)

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "MomentumTrendStrategy",
    "AdaptiveMomentumConfig",
    "PairsStrategy",
    "PairsConfig",
    "PortfolioMeanReversionStrategy",
    "PortfolioMeanReversionConfig",
    "StatisticalTrendStrategy",
    "StatisticalTrendConfig",
    "PMVMomentumStrategy",
    "PMVMomentumConfig",
    "NiftyShopStrategy",
    "NiftyShopConfig",
]
