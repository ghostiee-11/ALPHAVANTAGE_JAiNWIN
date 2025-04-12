import pandas as pd
import numpy as np
from enum import Enum

class TradeType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    REVERSE_LONG = "REVERSE_LONG"
    REVERSE_SHORT = "REVERSE_SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class Strategy:
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column: '{col}'")

        
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cum_vol_price'] / df['cum_vol']
        df['rsi'] = df['close'].rolling(14).apply(lambda x: 100 - 100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0)).sum())), raw=False)
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()

        df['trade_type'] = TradeType.HOLD.value
        df['TSL'] = np.nan
        df['SL'] = np.nan
        df['TP'] = np.nan

        position = None
        tsl = None
        sl = None
        atr_mult = 1.5
        reward_ratio = 2
        min_bb_width = 0.01  # Avoid sideways market

        for i in range(20, len(df)):
            price = df['close'][i]
            atr = df['atr'][i]
            bb_width = df['bb_width'][i]
            rsi = df['rsi'][i]
            ema_50 = df['ema_50'][i]
            ema_200 = df['ema_200'][i]

            long_signal = (ema_50 > ema_200 and price > df['vwap'][i] and bb_width > min_bb_width and rsi < 70)
            short_signal = (ema_50 < ema_200 and price < df['vwap'][i] and bb_width > min_bb_width and rsi > 30)

            
            hit_tsl = False
            if position == 'LONG' and price < tsl:
                df.at[i, 'trade_type'] = TradeType.CLOSE.value
                position = None
                tsl = None
                hit_tsl = True
            elif position == 'SHORT' and price > tsl:
                df.at[i, 'trade_type'] = TradeType.CLOSE.value
                position = None
                tsl = None
                hit_tsl = True

            if hit_tsl:
                continue

            
            if long_signal:
                if position is None:
                    df.at[i, 'trade_type'] = TradeType.LONG.value
                    position = 'LONG'
                    tsl = price - atr * atr_mult
                    sl = price - atr * atr_mult
                    df.at[i, 'SL'] = sl
                    df.at[i, 'TP'] = price + atr * atr_mult * reward_ratio
                elif position == 'SHORT':
                    df.at[i, 'trade_type'] = TradeType.REVERSE_LONG.value
                    position = 'LONG'
                    tsl = price - atr * atr_mult
                    sl = price - atr * atr_mult
                    df.at[i, 'SL'] = sl
                    df.at[i, 'TP'] = price + atr * atr_mult * reward_ratio
                else:
                    df.at[i, 'trade_type'] = TradeType.HOLD.value
                    tsl = max(tsl, price - atr * atr_mult)

            elif short_signal:
                if position is None:
                    df.at[i, 'trade_type'] = TradeType.SHORT.value
                    position = 'SHORT'
                    tsl = price + atr * atr_mult
                    sl = price + atr * atr_mult
                    df.at[i, 'SL'] = sl
                    df.at[i, 'TP'] = price - atr * atr_mult * reward_ratio
                elif position == 'LONG':
                    df.at[i, 'trade_type'] = TradeType.REVERSE_SHORT.value
                    position = 'SHORT'
                    tsl = price + atr * atr_mult
                    sl = price + atr * atr_mult
                    df.at[i, 'SL'] = sl
                    df.at[i, 'TP'] = price - atr * atr_mult * reward_ratio
                else:
                    df.at[i, 'trade_type'] = TradeType.HOLD.value
                    tsl = min(tsl, price + atr * atr_mult)

            
            elif position == 'LONG' and (price < df['ema_50'][i] or price < df['vwap'][i]):
                df.at[i, 'trade_type'] = TradeType.CLOSE.value
                position = None
                tsl = None

            elif position == 'SHORT' and (price > df['ema_50'][i] or price > df['vwap'][i]):
                df.at[i, 'trade_type'] = TradeType.CLOSE.value
                position = None
                tsl = None

            else:
                df.at[i, 'trade_type'] = TradeType.HOLD.value

            df.at[i, 'TSL'] = tsl

        return df