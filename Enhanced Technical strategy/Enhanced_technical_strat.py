import pandas as pd
import numpy as np
from enum import Enum
import pandas_ta as pta

class TradeType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    REVERSE_LONG = "REVERSE_LONG"
    REVERSE_SHORT = "REVERSE_SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class Strategy:
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the refined strategy aiming for better profitability.
        Changes: Wider SL, Stricter Entry (ADX/RSI), Simplified Exit, Optional No-Reversal.
        """
        if data.empty:
            data['trade_type'] = TradeType.HOLD.value
            data['TP'] = np.nan
            data['SL'] = np.nan
            data['TSL'] = np.nan
            return data

        df = data.copy()
        # Expecting columns: datetime, open, high, low, close, volume
        # Standardize known potential variations
        df.rename(columns={'Datetime': 'datetime', 'Open': 'open', 'High': 'high',
                           'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df.columns = [col.lower() for col in df.columns] # Ensure lowercase

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column: '{col}'")

        df.dropna(subset=required_cols, inplace=True)
        if len(df) < 200:
             print(f"Warning: Data length ({len(df)}) after NaN drop is less than 200. Results may be unreliable.")
             df['trade_type'] = TradeType.HOLD.value
             df['TP'] = np.nan
             df['SL'] = np.nan
             df['TSL'] = np.nan
             # Don't calculate indicators if data is too short
             return df

        # === Indicators using pandas_ta ===
        try:
            df.ta.ema(length=50, append=True, col_names=('ema_50',))
            df.ta.ema(length=200, append=True, col_names=('ema_200',))
            bbands_df = df.ta.bbands(length=20, std=2)
            # Assign directly using .loc to avoid potential SettingWithCopyWarning later
            df.loc[:, 'bb_lower'] = bbands_df['BBL_20_2.0']
            df.loc[:, 'bb_mid'] = bbands_df['BBM_20_2.0']
            df.loc[:, 'bb_upper'] = bbands_df['BBU_20_2.0']
            df.loc[:, 'bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, np.nan)

            df.loc[:, 'cum_vol_price'] = (df['close'] * df['volume']).cumsum()
            df.loc[:, 'cum_vol'] = df['volume'].cumsum()
            df.loc[:, 'vwap'] = df['cum_vol_price'] / df['cum_vol'].replace(0, np.nan)
            df['vwap'].ffill(inplace=True) # Handle potential initial NaNs if volume starts at 0

            df.ta.rsi(length=14, append=True, col_names=('rsi',))
            df.ta.atr(length=14, append=True, col_names=('atr',))
            adx_df = df.ta.adx(length=14)
            df.loc[:, 'adx'] = adx_df['ADX_14']
            df.loc[:, 'plus_di'] = adx_df['DMP_14']
            df.loc[:, 'minus_di'] = adx_df['DMN_14']
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            # Return df with available columns if indicators fail
            df['trade_type'] = TradeType.HOLD.value
            df['TP'] = np.nan
            df['SL'] = np.nan
            df['TSL'] = np.nan
            return df

        # === Strategy Logic ===
        df['trade_type'] = TradeType.HOLD.value
        df['TP'] = np.nan
        df['SL'] = np.nan
        df['TSL'] = np.nan

        # --- Parameters ---
        position = None # None, 'LONG', or 'SHORT'
        entry_price = np.nan
        tsl = np.nan
        sl = np.nan
        tp = np.nan

        # * Parameter Adjustments *
        atr_mult_sl = 2.5  # Widen SL (previously 1.5)
        atr_mult_tp = 3.0  # Aim for slightly higher Reward (previously 2.5)
        min_bb_width = 0.015 # Slightly higher min width? (optional)
        adx_threshold = 28   # Stricter ADX (previously 25)
        rsi_long_entry = 55  # Stricter RSI for long (previously 50)
        rsi_short_entry = 45 # Stricter RSI for short (previously 50)
        # adx_trend_weak_level = adx_threshold - 5 # Keep or remove? Let's simplify exit first

        # --- Option: Disable Direct Reversals ---
        allow_reversals = False # Set to True to allow REVERSE_LONG/SHORT, False to force CLOSE first

        first_valid_index = df.dropna(subset=['ema_200', 'bb_lower', 'rsi', 'atr', 'adx']).index.min()

        if pd.isna(first_valid_index):
            print("Warning: No valid starting index after indicator calculation.")
            return df # Already has default columns

        # --- Strategy Loop ---
        for i in df.index[df.index >= first_valid_index]:
            row = df.loc[i]
            # Check essential indicators for NaN again inside loop just in case
            essential_indicators = ['ema_50', 'ema_200', 'vwap', 'bb_width', 'adx', 'plus_di', 'minus_di', 'rsi', 'atr']
            if row[essential_indicators].isnull().any():
                df.loc[i, 'trade_type'] = TradeType.HOLD.value
                df.loc[i, 'SL'] = sl
                df.loc[i, 'TP'] = tp
                df.loc[i, 'TSL'] = tsl
                continue

            current_price = row['close']
            current_high = row['high']
            current_low = row['low']
            atr = row['atr']
            if atr == 0: atr = 0.0001 # Avoid division by zero or zero stops if ATR is momentarily 0

            adx = row['adx']
            plus_di = row['plus_di']
            minus_di = row['minus_di']
            rsi = row['rsi']
            ema_50 = row['ema_50']
            ema_200 = row['ema_200']
            vwap = row['vwap']
            bb_width = row['bb_width']

            exit_triggered = False
            exit_type = TradeType.CLOSE.value

            # --- Exit Logic ---
            if position == 'LONG':
                 # TSL update: Check TSL before SL/TP as it might have moved closer
                if current_low <= tsl:
                    exit_triggered = True
                elif current_low <= sl: # Check Initial SL
                    exit_triggered = True
                elif current_high >= tp: # Check TP
                    exit_triggered = True
                # Simplified Trend Weakness Exit: Only use EMA cross, more reliant on TSL/SL
                elif ema_50 < ema_200:
                     exit_triggered = True

                # Update TSL if position is still open
                if not exit_triggered:
                    potential_tsl = current_price - atr * atr_mult_sl
                    # Check if tsl is NaN (first bar after entry) or if potential_tsl is higher
                    if pd.isna(tsl) or potential_tsl > tsl:
                       tsl = potential_tsl # Move TSL up

            elif position == 'SHORT':
                if current_high >= tsl:
                    exit_triggered = True
                elif current_high >= sl: # Check Initial SL
                    exit_triggered = True
                elif current_low <= tp: # Check TP
                    exit_triggered = True
                # Simplified Trend Weakness Exit
                elif ema_50 > ema_200:
                     exit_triggered = True

                if not exit_triggered:
                    potential_tsl = current_price + atr * atr_mult_sl
                    if pd.isna(tsl) or potential_tsl < tsl:
                        tsl = potential_tsl # Move TSL down

            # --- Process Exit ---
            if exit_triggered:
                df.loc[i, 'trade_type'] = exit_type # Set to CLOSE
                position = None
                entry_price = np.nan
                df.loc[i, 'SL'] = sl
                df.loc[i, 'TP'] = tp
                df.loc[i, 'TSL'] = tsl # Record the TSL value at exit
                sl, tp, tsl = np.nan, np.nan, np.nan # Reset internal state
                continue

            # --- Entry / Reversal Logic ---
            trade_executed_this_bar = False

            long_signal = (
                ema_50 > ema_200
                and current_price > vwap # Consider adding a small buffer? e.g., current_price > vwap * 1.001
                and bb_width > min_bb_width
                and adx > adx_threshold
                and plus_di > minus_di
                and rsi >= rsi_long_entry
            )

            short_signal = (
                ema_50 < ema_200
                and current_price < vwap # Consider buffer? e.g., current_price < vwap * 0.999
                and bb_width > min_bb_width
                and adx > adx_threshold
                and minus_di > plus_di
                and rsi <= rsi_short_entry
            )

            # --- Execute Trades ---
            if position is None: # Only new entries if flat
                if long_signal:
                    df.loc[i, 'trade_type'] = TradeType.LONG.value
                    position = 'LONG'
                    entry_price = current_price
                    sl = entry_price - atr * atr_mult_sl
                    # TSL starts at SL level, will be updated on next bars if price moves favorably
                    tsl = sl
                    tp = entry_price + atr * atr_mult_tp
                    trade_executed_this_bar = True
                elif short_signal:
                    df.loc[i, 'trade_type'] = TradeType.SHORT.value
                    position = 'SHORT'
                    entry_price = current_price
                    sl = entry_price + atr * atr_mult_sl
                    tsl = sl
                    tp = entry_price - atr * atr_mult_tp
                    trade_executed_this_bar = True

            elif allow_reversals: # Handle reversals only if allowed
                if position == 'SHORT' and long_signal:
                    df.loc[i, 'trade_type'] = TradeType.REVERSE_LONG.value
                    position = 'LONG'
                    entry_price = current_price
                    sl = entry_price - atr * atr_mult_sl
                    tsl = sl
                    tp = entry_price + atr * atr_mult_tp
                    trade_executed_this_bar = True
                elif position == 'LONG' and short_signal:
                    df.loc[i, 'trade_type'] = TradeType.REVERSE_SHORT.value
                    position = 'SHORT'
                    entry_price = current_price
                    sl = entry_price + atr * atr_mult_sl
                    tsl = sl
                    tp = entry_price - atr * atr_mult_tp
                    trade_executed_this_bar = True

            # --- Set HOLD if no action ---
            if not trade_executed_this_bar:
                 df.loc[i, 'trade_type'] = TradeType.HOLD.value

            # --- Store State ---
            df.loc[i, 'SL'] = sl
            df.loc[i, 'TP'] = tp
            df.loc[i, 'TSL'] = tsl

        # Optional: Clean up intermediate columns
        # df.drop(columns=['cum_vol_price', 'cum_vol', 'plus_di', 'minus_di'], inplace=True, errors='ignore')

        return df