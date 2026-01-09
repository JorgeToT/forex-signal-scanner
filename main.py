# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 01:17:11 2026

@author: eduar
"""

# -*- coding: utf-8 -*-
"""
Multi-timeframe Forex signal (4H, 1H, 30m) - EDUCATIONAL ONLY.
Includes:
- BUY/SELL/HOLD decision (4H/1H/30m)
- Confidence score
- SL/TP from ATR, optionally snapped to Fibonacci levels
- Position sizing (lotaje) with fixed % risk (assumes USD account)
- Text summary of what to do

Data source: Yahoo Finance via yfinance (quotes may differ from broker).
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import csv
import os
import pandas as pd
import yfinance as yf


# ----------------------------
# Indicators
# ----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# ----------------------------
# Spread típico por par (en pips)
# Estos son spreads promedio de brokers ECN/STP en condiciones normales
# Ajusta según tu broker específico
# ----------------------------
TYPICAL_SPREADS = {
    # Majors (spreads más bajos)
    "EURUSD": 1.0,
    "USDJPY": 1.0,
    "GBPUSD": 1.2,
    "USDCHF": 1.3,
    "AUDUSD": 1.2,
    "USDCAD": 1.5,
    "NZDUSD": 1.5,
    
    # Crosses (spreads más altos)
    "EURGBP": 1.5,
    "EURJPY": 1.5,
    "GBPJPY": 2.5,
    "AUDJPY": 2.0,
    "EURAUD": 2.0,
    "EURCHF": 2.0,
    "GBPCHF": 3.0,
    "AUDNZD": 2.5,
    "AUDCAD": 2.0,
    "NZDJPY": 2.5,
    "CADJPY": 2.0,
}


def get_spread_adjustment(ticker: str) -> Dict[str, float]:
    """
    Retorna el ajuste de spread para un par de divisas.
    
    En Forex:
    - BUY: Entras al ASK, cierras (TP/SL) al BID
      → SL efectivo debe estar más abajo (restar buffer)
      → TP efectivo debe estar más arriba (sumar buffer)
    
    - SELL: Entras al BID, cierras (TP/SL) al ASK
      → SL efectivo debe estar más arriba (sumar buffer)
      → TP efectivo debe estar más abajo (restar buffer)
    
    Retorna spread en unidades de precio (no pips).
    """
    symbol = ticker.replace("=X", "").upper()
    spread_pips = TYPICAL_SPREADS.get(symbol, 2.0)  # Default 2 pips
    
    # Convertir pips a precio
    pip = 0.01 if "JPY" in symbol else 0.0001
    spread_price = spread_pips * pip
    
    return {
        "symbol": symbol,
        "spread_pips": spread_pips,
        "spread_price": spread_price,
        "pip": pip
    }


def adjust_sl_tp_for_spread(
    entry: float,
    sl: float,
    tp: float,
    action: str,
    ticker: str,
    buffer_multiplier: float = 1.0
) -> Dict[str, float]:
    """
    Ajusta SL y TP para compensar el spread.
    
    Parámetros:
    - buffer_multiplier: 1.0 = spread exacto, 1.5 = 50% extra de seguridad
    
    Retorna:
    - sl_adjusted: SL ajustado por spread
    - tp_adjusted: TP ajustado por spread
    - sl_pips_added: Pips agregados al SL
    - tp_pips_added: Pips agregados al TP
    """
    spread_info = get_spread_adjustment(ticker)
    spread = spread_info["spread_price"] * buffer_multiplier
    pip = spread_info["pip"]
    
    if action == "BUY":
        # BUY: SL se activa cuando BID cae → mover SL más abajo
        # TP se activa cuando BID sube → mover TP más arriba
        sl_adjusted = sl - spread
        tp_adjusted = tp + spread
    elif action == "SELL":
        # SELL: SL se activa cuando ASK sube → mover SL más arriba
        # TP se activa cuando ASK baja → mover TP más abajo
        sl_adjusted = sl + spread
        tp_adjusted = tp - spread
    else:
        return {"sl_adjusted": sl, "tp_adjusted": tp, "spread_pips": 0}
    
    return {
        "sl_original": sl,
        "tp_original": tp,
        "sl_adjusted": sl_adjusted,
        "tp_adjusted": tp_adjusted,
        "spread_pips": spread_info["spread_pips"],
        "sl_buffer_pips": round(spread / pip, 1),
        "tp_buffer_pips": round(spread / pip, 1),
        "description": f"📊 Spread ~{spread_info['spread_pips']:.1f} pips → SL/TP ajustados"
    }


def detect_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 20) -> Dict[str, object]:
    """
    Detecta divergencias entre precio y RSI.
    
    Divergencia Alcista (bullish): 
        - Precio hace Lower Low pero RSI hace Higher Low
        - Señal de posible reversión alcista
    
    Divergencia Bajista (bearish):
        - Precio hace Higher High pero RSI hace Lower High  
        - Señal de posible reversión bajista
    
    Retorna: {"type": "bullish"|"bearish"|None, "strength": 0-1, "description": str}
    """
    if len(close) < lookback + 5:
        return {"type": None, "strength": 0.0, "description": "Datos insuficientes"}
    
    # Tomar últimas N velas
    price = close.iloc[-lookback:].values
    rsi_vals = rsi_series.iloc[-lookback:].values
    
    # Encontrar pivots (mínimos y máximos locales)
    def find_pivots(data, order=3):
        """Encuentra índices de pivots (mínimos y máximos locales)."""
        highs = []
        lows = []
        for i in range(order, len(data) - order):
            # Pivot high
            if all(data[i] > data[i-j] for j in range(1, order+1)) and \
               all(data[i] > data[i+j] for j in range(1, order+1)):
                highs.append(i)
            # Pivot low
            if all(data[i] < data[i-j] for j in range(1, order+1)) and \
               all(data[i] < data[i+j] for j in range(1, order+1)):
                lows.append(i)
        return highs, lows
    
    price_highs, price_lows = find_pivots(price, order=2)
    rsi_highs, rsi_lows = find_pivots(rsi_vals, order=2)
    
    divergence = {"type": None, "strength": 0.0, "description": "Sin divergencia"}
    
    # Buscar Divergencia Alcista (bullish)
    # Necesitamos al menos 2 lows en precio y RSI
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        # Comparar los últimos 2 lows
        last_price_lows = sorted(price_lows)[-2:]
        last_rsi_lows = sorted(rsi_lows)[-2:]
        
        # Precio hace lower low
        price_lower_low = price[last_price_lows[-1]] < price[last_price_lows[-2]]
        # RSI hace higher low
        rsi_higher_low = rsi_vals[last_rsi_lows[-1]] > rsi_vals[last_rsi_lows[-2]]
        
        if price_lower_low and rsi_higher_low:
            # Calcular fuerza basada en la diferencia
            rsi_diff = rsi_vals[last_rsi_lows[-1]] - rsi_vals[last_rsi_lows[-2]]
            strength = min(abs(rsi_diff) / 10, 1.0)  # Normalizar a 0-1
            
            divergence = {
                "type": "bullish",
                "strength": round(strength, 2),
                "description": f"📈 Divergencia ALCISTA: Precio ↓ pero RSI ↑ (fuerza: {strength:.0%})"
            }
    
    # Buscar Divergencia Bajista (bearish) - solo si no encontramos alcista
    if divergence["type"] is None and len(price_highs) >= 2 and len(rsi_highs) >= 2:
        last_price_highs = sorted(price_highs)[-2:]
        last_rsi_highs = sorted(rsi_highs)[-2:]
        
        # Precio hace higher high
        price_higher_high = price[last_price_highs[-1]] > price[last_price_highs[-2]]
        # RSI hace lower high
        rsi_lower_high = rsi_vals[last_rsi_highs[-1]] < rsi_vals[last_rsi_highs[-2]]
        
        if price_higher_high and rsi_lower_high:
            rsi_diff = rsi_vals[last_rsi_highs[-2]] - rsi_vals[last_rsi_highs[-1]]
            strength = min(abs(rsi_diff) / 10, 1.0)
            
            divergence = {
                "type": "bearish",
                "strength": round(strength, 2),
                "description": f"📉 Divergencia BAJISTA: Precio ↑ pero RSI ↓ (fuerza: {strength:.0%})"
            }
    
    return divergence


def calculate_adr(ticker: str, period: int = 14) -> Dict[str, object]:
    """
    Calcula el Average Daily Range (ADR) y el porcentaje del movimiento de hoy.
    
    ADR = Promedio del rango (High - Low) de los últimos N días
    
    Retorna:
    - adr: Rango promedio diario en precio
    - adr_pips: ADR en pips
    - today_range: Rango de hoy hasta ahora
    - today_pct: % del ADR que ya se movió hoy
    - exhausted: True si hoy ya se movió >70% del ADR
    """
    try:
        # Obtener datos diarios
        df = yf.download(
            tickers=ticker,
            period=f"{period + 5}d",
            interval="1d",
            auto_adjust=False,
            progress=False
        )
        
        if df is None or len(df) < period:
            return {"adr": None, "exhausted": False, "description": "Datos insuficientes"}
        
        df = _normalize_yfinance_columns(df)
        
        # Calcular ADR (promedio del rango diario)
        df["Range"] = df["High"] - df["Low"]
        adr = float(df["Range"].iloc[-(period+1):-1].mean())  # Excluir hoy
        
        # Rango de hoy
        today_high = float(df["High"].iloc[-1])
        today_low = float(df["Low"].iloc[-1])
        today_range = today_high - today_low
        
        # Porcentaje del ADR usado hoy
        today_pct = (today_range / adr * 100) if adr > 0 else 0
        
        # Determinar si el movimiento está agotado
        exhausted = today_pct >= 70
        
        # Calcular en pips
        pip = 0.01 if "JPY" in ticker.upper() else 0.0001
        adr_pips = adr / pip
        today_range_pips = today_range / pip
        remaining_pips = max(0, adr_pips - today_range_pips)
        
        if exhausted:
            description = f"⚠️ Movimiento AGOTADO: {today_pct:.0f}% del ADR usado ({today_range_pips:.0f}/{adr_pips:.0f} pips)"
        elif today_pct >= 50:
            description = f"🟡 Movimiento avanzado: {today_pct:.0f}% del ADR ({remaining_pips:.0f} pips restantes)"
        else:
            description = f"🟢 Movimiento fresco: {today_pct:.0f}% del ADR ({remaining_pips:.0f} pips disponibles)"
        
        return {
            "adr": adr,
            "adr_pips": round(adr_pips, 1),
            "today_range": today_range,
            "today_range_pips": round(today_range_pips, 1),
            "today_pct": round(today_pct, 1),
            "remaining_pips": round(remaining_pips, 1),
            "exhausted": exhausted,
            "description": description
        }
        
    except Exception as e:
        return {"adr": None, "exhausted": False, "description": f"Error: {str(e)[:30]}"}


def detect_liquidity_zones(
    ticker: str,
    timeframe: str = "4h",
    lookback_bars: int = 100,
    atr_multiplier: float = 0.5
) -> Dict[str, object]:
    """
    Detecta zonas de liquidez basadas en swing highs/lows donde se acumulan stops.
    
    La liquidez se acumula en:
    - Por encima de swing highs (stops de posiciones SHORT)
    - Por debajo de swing lows (stops de posiciones LONG)
    
    Estas zonas son "imanes" para el precio antes de reversiones significativas.
    
    Parámetros:
    - lookback_bars: Cuántas velas analizar
    - atr_multiplier: Para agrupar zonas cercanas (% del ATR)
    
    Retorna:
    - liquidity_above: Lista de zonas de liquidez por encima del precio
    - liquidity_below: Lista de zonas de liquidez por debajo del precio
    - nearest_above/below: Zona más cercana en cada dirección
    - description: Resumen
    """
    try:
        # Obtener datos
        df = yf.download(
            tickers=ticker,
            period="60d",
            interval=timeframe,
            auto_adjust=False,
            progress=False
        )
        
        if df is None or len(df) < lookback_bars:
            return {"description": "Datos insuficientes para detectar liquidez"}
        
        df = _normalize_yfinance_columns(df)
        df = df.iloc[-lookback_bars:]
        
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        current_price = float(closes[-1])
        
        # Calcular ATR para agrupar zonas
        tr = []
        for i in range(1, len(df)):
            h = highs[i]
            l = lows[i]
            c_prev = closes[i-1]
            tr.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
        atr = sum(tr[-14:]) / 14 if len(tr) >= 14 else sum(tr) / len(tr) if tr else 0
        zone_tolerance = atr * atr_multiplier
        
        pip = 0.01 if "JPY" in ticker.upper() else 0.0001
        
        # Detectar swing highs (liquidez arriba - stops de SHORTS)
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
        
        # Detectar swing lows (liquidez abajo - stops de LONGS)
        swing_lows = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        # Agrupar zonas cercanas y contar "fortaleza" (cuántos swings en la zona)
        def cluster_zones(levels, tolerance):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if level - current_cluster[-1] <= tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append({
                        "price": sum(current_cluster) / len(current_cluster),
                        "strength": len(current_cluster),
                        "touches": len(current_cluster)
                    })
                    current_cluster = [level]
            
            clusters.append({
                "price": sum(current_cluster) / len(current_cluster),
                "strength": len(current_cluster),
                "touches": len(current_cluster)
            })
            return clusters
        
        liquidity_above = [z for z in cluster_zones(swing_highs, zone_tolerance) 
                          if z["price"] > current_price]
        liquidity_below = [z for z in cluster_zones(swing_lows, zone_tolerance) 
                          if z["price"] < current_price]
        
        # Ordenar por cercanía al precio
        liquidity_above.sort(key=lambda x: x["price"])
        liquidity_below.sort(key=lambda x: x["price"], reverse=True)
        
        # Zona más cercana
        nearest_above = liquidity_above[0] if liquidity_above else None
        nearest_below = liquidity_below[0] if liquidity_below else None
        
        decimals = 3 if "JPY" in ticker.upper() else 5
        
        # Construir descripción
        desc_lines = ["🎯 ZONAS DE LIQUIDEZ:"]
        
        if nearest_above:
            dist_pips = abs(nearest_above["price"] - current_price) / pip
            strength = "🔥" * min(nearest_above["strength"], 3)
            desc_lines.append(
                f"   ↑ ARRIBA: {nearest_above['price']:.{decimals}f} ({dist_pips:.0f} pips) {strength}"
            )
        
        if nearest_below:
            dist_pips = abs(current_price - nearest_below["price"]) / pip
            strength = "🔥" * min(nearest_below["strength"], 3)
            desc_lines.append(
                f"   ↓ ABAJO:  {nearest_below['price']:.{decimals}f} ({dist_pips:.0f} pips) {strength}"
            )
        
        if not nearest_above and not nearest_below:
            desc_lines.append("   No se detectaron zonas claras")
        
        return {
            "liquidity_above": liquidity_above[:3],  # Top 3
            "liquidity_below": liquidity_below[:3],
            "nearest_above": nearest_above,
            "nearest_below": nearest_below,
            "current_price": current_price,
            "description": "\n".join(desc_lines)
        }
        
    except Exception as e:
        return {"description": f"Error detectando liquidez: {str(e)[:30]}"}


def calculate_quality_score(signal: dict) -> Dict[str, object]:
    """
    Calcula un Score de Calidad (A, B, C) basado en múltiples factores.
    
    Criterios evaluados:
    - Confianza del modelo (peso: 25%)
    - Ratio R:R (peso: 20%)
    - Alineación multi-timeframe (peso: 20%)
    - Divergencia RSI presente (peso: 15%)
    - ADR disponible (peso: 10%)
    - Sesión óptima (peso: 10%)
    
    Score final:
    - A: >= 80 puntos (Setup de alta probabilidad)
    - B: >= 60 puntos (Setup aceptable)
    - C: < 60 puntos (Setup de baja calidad)
    
    Retorna:
    - grade: "A", "B", o "C"
    - score: Puntuación numérica (0-100)
    - breakdown: Desglose de puntos por categoría
    - description: Resumen visual
    """
    breakdown = {}
    total_score = 0
    
    # 1. Confianza del modelo (25 puntos máx)
    conf = signal.get("final_confidence", 0)
    conf_score = min(conf * 25, 25)  # 0-1 → 0-25
    breakdown["confianza"] = round(conf_score, 1)
    total_score += conf_score
    
    # 2. Ratio R:R (20 puntos máx)
    rk = signal.get("risk", {})
    entry = rk.get("entry", 0)
    sl = rk.get("sl", 0)
    tp = rk.get("tp", 0)
    
    if entry and sl and tp:
        sl_dist = abs(entry - sl)
        tp_dist = abs(entry - tp)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        
        # R:R >= 2.5 = 20pts, R:R >= 2.0 = 16pts, R:R >= 1.5 = 12pts, else escalar
        if rr >= 2.5:
            rr_score = 20
        elif rr >= 2.0:
            rr_score = 16
        elif rr >= 1.5:
            rr_score = 12
        else:
            rr_score = max(0, rr * 6)
    else:
        rr_score = 0
    breakdown["rr_ratio"] = round(rr_score, 1)
    total_score += rr_score
    
    # 3. Alineación multi-timeframe (20 puntos máx)
    # Checar si los 3 TFs apuntan en la misma dirección
    tfs = signal.get("timeframes", {})
    action = signal.get("summary", {}).get("action", "HOLD")
    
    aligned_count = 0
    for tf_name, tf_data in tfs.items():
        tf_score = tf_data.get("score", 0)
        if action == "BUY" and tf_score > 0:
            aligned_count += 1
        elif action == "SELL" and tf_score < 0:
            aligned_count += 1
    
    # 3 TFs alineados = 20pts, 2 = 14pts, 1 = 7pts
    mtf_score = aligned_count * 7 if aligned_count < 3 else 20
    breakdown["alineacion_mtf"] = round(mtf_score, 1)
    total_score += mtf_score
    
    # 4. Divergencia RSI (15 puntos máx)
    # Bonus si hay divergencia que confirma la dirección
    div_score = 0
    for tf_name, tf_data in tfs.items():
        div = tf_data.get("divergence", {})
        if div and div.get("type"):
            div_type = div.get("type")
            div_strength = div.get("strength", 0)
            
            # Divergencia alcista + BUY o Divergencia bajista + SELL = confirma
            if (div_type == "bullish" and action == "BUY") or \
               (div_type == "bearish" and action == "SELL"):
                div_score = max(div_score, 10 + (div_strength * 5))
    
    breakdown["divergencia"] = round(min(div_score, 15), 1)
    total_score += min(div_score, 15)
    
    # 5. ADR disponible (10 puntos máx)
    adr_info = signal.get("adr", {})
    adr_pct = adr_info.get("today_pct", 100)
    
    # Menos ADR usado = mejor. <30% = 10pts, <50% = 7pts, <70% = 4pts
    if adr_pct < 30:
        adr_score = 10
    elif adr_pct < 50:
        adr_score = 7
    elif adr_pct < 70:
        adr_score = 4
    else:
        adr_score = 0
    breakdown["adr_disponible"] = round(adr_score, 1)
    total_score += adr_score
    
    # 6. Sesión óptima (10 puntos máx)
    session_optimal = signal.get("session_optimal", False)
    session_score = 10 if session_optimal else 3
    breakdown["sesion_optima"] = round(session_score, 1)
    total_score += session_score
    
    # Determinar grado
    total_score = round(total_score, 1)
    if total_score >= 80:
        grade = "A"
        grade_emoji = "🅰️"
        grade_desc = "ALTA PROBABILIDAD"
    elif total_score >= 60:
        grade = "B"
        grade_emoji = "🅱️"
        grade_desc = "SETUP ACEPTABLE"
    else:
        grade = "C"
        grade_emoji = "©️"
        grade_desc = "BAJA CALIDAD"
    
    # Construir descripción
    desc_lines = [
        f"📊 CALIDAD DEL SETUP: {grade_emoji} {grade} ({total_score}/100) - {grade_desc}",
        f"   ├─ Confianza:     {breakdown['confianza']:.0f}/25",
        f"   ├─ R:R Ratio:     {breakdown['rr_ratio']:.0f}/20",
        f"   ├─ Alineación TF: {breakdown['alineacion_mtf']:.0f}/20",
        f"   ├─ Divergencia:   {breakdown['divergencia']:.0f}/15",
        f"   ├─ ADR Disp:      {breakdown['adr_disponible']:.0f}/10",
        f"   └─ Sesión:        {breakdown['sesion_optima']:.0f}/10",
    ]
    
    return {
        "grade": grade,
        "score": total_score,
        "breakdown": breakdown,
        "description": "\n".join(desc_lines)
    }


# ----------------------------
# Utils
# ----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns (if any) and ensure OHLC exists."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    needed = {"Open", "High", "Low", "Close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLC: {missing}. Columnas recibidas: {list(df.columns)}")
    return df

def fetch_ohlc(ticker: str, interval: str, lookback: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=lookback,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True
    )
    df = _normalize_yfinance_columns(df)
    df = df.dropna()
    if df.empty:
        raise ValueError(f"No se pudieron obtener datos para {ticker} con intervalo {interval}.")
    return df


# ----------------------------
# Fibonacci
# ----------------------------
FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXT = [1.272, 1.618]

def compute_fib_levels(df: pd.DataFrame, bars: int = 120) -> Dict[str, object]:
    """Compute fib levels from swing high/low in last `bars` candles."""
    if len(df) < bars + 5:
        bars = max(30, len(df) - 5)

    window = df.tail(bars).copy()
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())

    idx_high = window["High"].idxmax()
    idx_low = window["Low"].idxmin()

    # Use positions to avoid ambiguity
    pos_high = int(window.index.get_loc(idx_high))
    pos_low = int(window.index.get_loc(idx_low))
    direction = "UP" if pos_low < pos_high else "DOWN"

    if swing_high == swing_low:
        return {"direction": "FLAT", "swing_high": swing_high, "swing_low": swing_low, "levels_sorted": []}

    levels = {}
    if direction == "UP":
        diff = swing_high - swing_low
        levels["0.0"] = swing_low
        levels["1.0"] = swing_high
        for r in FIB_RATIOS:
            levels[str(r)] = swing_high - diff * r
        for e in FIB_EXT:
            levels[str(e)] = swing_high + diff * (e - 1.0)
    else:
        diff = swing_high - swing_low
        levels["0.0"] = swing_high
        levels["1.0"] = swing_low
        for r in FIB_RATIOS:
            levels[str(r)] = swing_low + diff * r
        for e in FIB_EXT:
            levels[str(e)] = swing_low - diff * (e - 1.0)

    levels_sorted = sorted([(k, float(v)) for k, v in levels.items()], key=lambda x: x[1])
    return {
        "direction": direction,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "levels_sorted": levels_sorted
    }

def nearest_fib_above(levels_sorted: List[Tuple[str, float]], price: float) -> Optional[Tuple[str, float]]:
    for k, v in levels_sorted:
        if v > price:
            return (k, v)
    return None

def nearest_fib_below(levels_sorted: List[Tuple[str, float]], price: float) -> Optional[Tuple[str, float]]:
    for k, v in reversed(levels_sorted):
        if v < price:
            return (k, v)
    return None


# ----------------------------
# Position sizing (Lotaje)
# ----------------------------
def pip_size_for_ticker(ticker: str) -> float:
    return 0.01 if "JPY" in ticker.upper() else 0.0001

def base_quote_from_ticker(ticker: str) -> tuple:
    t = ticker.replace("=X", "").upper()
    if len(t) != 6:
        return ("", "")
    return (t[:3], t[3:])

def get_usdjpy_rate() -> float:
    """Obtiene el rate actual de USDJPY para conversiones."""
    try:
        df = yf.download("USDJPY=X", period="1d", interval="1h", progress=False)
        if df is not None and not df.empty:
            df = _normalize_yfinance_columns(df)
            return float(df["Close"].iloc[-1])
    except:
        pass
    return 150.0  # Fallback razonable


def pip_value_per_1lot_usd(ticker: str, price: float, usdjpy_rate: float = None) -> float:
    """
    Pip value in USD for 1 standard lot (100,000 units),
    assuming account currency = USD.
    
    - Quote USD (EURUSD, GBPUSD): $10 / pip / lot exacto
    - Base USD (USDJPY): (pip * 100000) / USDJPY rate  
    - Cross XXX/JPY: (pip * 100000) / USDJPY rate  
    - Cross XXX/YYY: aproximación, idealmente convertir YYY->USD
    """
    base, quote = base_quote_from_ticker(ticker)
    pip = pip_size_for_ticker(ticker)
    contract = 100_000

    # Si quote es USD -> pip value es exactamente $10 (para 0.0001) o $1000 para JPY crosses cotizados en USD
    if quote == "USD":
        return pip * contract  # = $10 para pares como EURUSD
    
    # Si base es USD (ej: USDJPY, USDCAD, USDCHF)
    if base == "USD":
        return (pip * contract) / price
    
    # Cross pairs con JPY como quote (EURJPY, GBPJPY, CADJPY, etc.)
    # Pip value = (0.01 * 100000) / USDJPY = 1000 / USDJPY
    if quote == "JPY":
        # Usamos el rate proporcionado o estimamos ~150 como fallback
        jpy_rate = usdjpy_rate if usdjpy_rate else 150.0
        return (pip * contract) / jpy_rate
    
    # Otros crosses (EURGBP, AUDNZD, etc.) - aproximación
    # Idealmente habría que convertir la quote currency a USD
    # Por ahora usamos el precio del par como proxy
    return (pip * contract) / price if price > 1 else pip * contract

def position_size_lots(
    ticker: str,
    balance_usd: float,
    risk_pct: float,
    entry: float,
    sl: float,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
    usdjpy_rate: float = None
) -> dict:
    """
    Fixed-% risk sizing.
    El SL es fijo (viene del análisis técnico).
    El lotaje se ajusta para que el riesgo sea exactamente risk_pct del balance.
    
    Fórmula: Lotes = Riesgo_USD / (Stop_Pips × Pip_Value_por_Lote)
    """
    if sl is None or entry is None:
        return {"lots": None, "reason": "No SL/entry."}

    risk_usd = balance_usd * risk_pct
    pip = pip_size_for_ticker(ticker)
    stop_pips = abs(entry - sl) / pip
    
    if stop_pips <= 0:
        return {"lots": None, "reason": "Stop pips invalid."}

    # Obtener USDJPY rate si es necesario para crosses JPY
    base, quote = base_quote_from_ticker(ticker)
    if quote == "JPY" and base != "USD" and usdjpy_rate is None:
        usdjpy_rate = get_usdjpy_rate()

    pip_value_1lot = pip_value_per_1lot_usd(ticker, entry, usdjpy_rate)
    
    # Calcular lotes necesarios para el riesgo deseado
    raw_lots = risk_usd / (stop_pips * pip_value_1lot)

    # Redondear hacia abajo al step
    lots = max(0.0, raw_lots)
    lots = (lots // lot_step) * lot_step
    if lots < min_lot:
        lots = min_lot

    # Riesgo real con lotes redondeados
    actual_risk = stop_pips * pip_value_1lot * lots

    return {
        "balance_usd": balance_usd,
        "risk_pct": risk_pct,
        "risk_usd_target": risk_usd,
        "pip_size": pip,
        "stop_pips": stop_pips,
        "pip_value_per_1lot_usd": pip_value_1lot,
        "lots_raw": raw_lots,
        "lots": lots,
        "units": lots * 100_000,
        "risk_usd_actual": actual_risk,
        "usdjpy_rate_used": usdjpy_rate if quote == "JPY" and base != "USD" else None
    }


# ----------------------------
# Strategy
# ----------------------------
@dataclass
class TFConfig:
    interval: str
    lookback: str
    weight: float

TF_MAP = {
    "4H":  TFConfig(interval="4h",  lookback="90d",  weight=0.5),
    "1H":  TFConfig(interval="1h",  lookback="60d",  weight=0.3),
    "30m": TFConfig(interval="30m", lookback="30d",  weight=0.2),
}

def score_timeframe(df: pd.DataFrame) -> Dict[str, float]:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema_fast = ema(close, 21)
    ema_slow = ema(close, 55)
    r = rsi(close, 14)
    _, _, hist = macd(close)
    a = atr(high, low, close, 14)

    last = df.index[-1]
    trend_up = float(ema_fast.loc[last] > ema_slow.loc[last])
    trend_dn = float(ema_fast.loc[last] < ema_slow.loc[last])

    rsi_val = float(r.loc[last])
    hist_val = float(hist.loc[last])
    atr_val = float(a.loc[last])
    close_val = float(close.loc[last])

    atr_pct = (atr_val / close_val) * 100.0

    # Detectar divergencias RSI
    divergence = detect_divergence(close, r, lookback=25)

    score = 0.0
    if trend_up:
        score += 0.45
    elif trend_dn:
        score -= 0.45

    if 55 <= rsi_val <= 75:
        score += 0.25
    elif 25 <= rsi_val <= 45:
        score -= 0.25

    if hist_val > 0:
        score += 0.20
    elif hist_val < 0:
        score -= 0.20

    # Bonus/Penalización por divergencia
    # Divergencia alcista en tendencia bajista = señal de reversión (reduce sell score)
    # Divergencia bajista en tendencia alcista = señal de reversión (reduce buy score)
    divergence_impact = 0.0
    if divergence["type"] == "bullish":
        # Favorece compra / penaliza venta
        divergence_impact = divergence["strength"] * 0.25
        score += divergence_impact
    elif divergence["type"] == "bearish":
        # Favorece venta / penaliza compra
        divergence_impact = divergence["strength"] * 0.25
        score -= divergence_impact

    vol_penalty = 0.20 if atr_pct < 0.08 else 0.0

    score = clamp(score, -1.0, 1.0)
    confidence = clamp(abs(score) - vol_penalty, 0.0, 1.0)
    
    # Bonus de confianza si hay divergencia confirmando la dirección
    if (divergence["type"] == "bullish" and score > 0) or \
       (divergence["type"] == "bearish" and score < 0):
        confidence = clamp(confidence + divergence["strength"] * 0.15, 0.0, 1.0)

    return {
        "score": score,
        "confidence": confidence,
        "close": float(close_val),
        "rsi": rsi_val,
        "macd_hist": hist_val,
        "atr_pct": atr_pct,
        "atr": atr_val,
        "divergence": divergence
    }

def aggregate_signal(tf_results: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    total_w = sum(TF_MAP[tf].weight for tf in tf_results)
    w_score = sum(tf_results[tf]["score"] * TF_MAP[tf].weight for tf in tf_results) / (total_w or 1.0)

    signs = [1 if tf_results[tf]["score"] > 0 else (-1 if tf_results[tf]["score"] < 0 else 0) for tf in tf_results]
    agreement = (all(s > 0 for s in signs) or all(s < 0 for s in signs))

    action = "HOLD"
    if w_score >= 0.20:
        action = "BUY"
    elif w_score <= -0.20:
        action = "SELL"

    if agreement and action != "HOLD":
        w_score = clamp(w_score * 1.10, -1.0, 1.0)

    return {"action": action, "weighted_score": w_score, "agreement": agreement}

def compute_final_confidence(weighted_score: float, agreement: bool, tf_results: dict) -> float:
    base = abs(weighted_score)
    bonus = 0.10 if agreement else 0.0
    atr30 = tf_results.get("30m", {}).get("atr_pct", 0.0)
    penalty = 0.15 if atr30 < 0.06 else 0.0
    return clamp(base + bonus - penalty, 0.0, 1.0)

def compute_sl_tp_atr(action: str, entry: float, atr_value: float, rr: float = 2.0, atr_mult_sl: float = 1.5) -> dict:
    sl_dist = atr_mult_sl * atr_value
    tp_dist = rr * sl_dist

    if action == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
    elif action == "SELL":
        sl = entry + sl_dist
        tp = entry - tp_dist
    else:
        sl, tp = None, None

    return {
        "method": "ATR",
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "sl_distance": sl_dist,
        "tp_distance": tp_dist,
        "rr": rr,
        "atr_mult_sl": atr_mult_sl,
        "atr_used": atr_value
    }

def adjust_sl_tp_with_fib(
    action: str,
    entry: float,
    atr_sl: float,
    atr_tp: float,
    fib: Dict[str, object],
    max_adjust_pct_of_atr: float = 0.75
) -> dict:
    levels_sorted = fib.get("levels_sorted", [])
    if action == "HOLD":
        return {"method": "FIB+ATR", "entry": entry, "sl": None, "tp": None, "notes": "HOLD"}
    if not levels_sorted:
        return {"method": "FIB+ATR", "entry": entry, "sl": atr_sl, "tp": atr_tp, "sl_tag": "ATR", "tp_tag": "ATR"}

    atr_dist_sl = abs(entry - atr_sl)
    atr_dist_tp = abs(entry - atr_tp)
    max_sl_adjust = max_adjust_pct_of_atr * atr_dist_sl
    max_tp_adjust = max_adjust_pct_of_atr * atr_dist_tp

    sl_final, tp_final = atr_sl, atr_tp
    sl_tag, tp_tag = "ATR", "ATR"

    if action == "BUY":
        cand_tp = nearest_fib_above(levels_sorted, entry)
        cand_sl = nearest_fib_below(levels_sorted, entry)
    else:
        cand_tp = nearest_fib_below(levels_sorted, entry)
        cand_sl = nearest_fib_above(levels_sorted, entry)

    if cand_tp is not None:
        _, fib_tp = cand_tp
        if abs(fib_tp - atr_tp) <= max_tp_adjust:
            tp_final = fib_tp
            tp_tag = f"FIB({cand_tp[0]})"

    if cand_sl is not None:
        _, fib_sl = cand_sl
        if abs(fib_sl - atr_sl) <= max_sl_adjust:
            sl_final = fib_sl
            sl_tag = f"FIB({cand_sl[0]})"

    return {
        "method": "FIB+ATR",
        "entry": entry,
        "sl": sl_final,
        "tp": tp_final,
        "sl_tag": sl_tag,
        "tp_tag": tp_tag,
        "fib_direction": fib.get("direction"),
        "swing_high": fib.get("swing_high"),
        "swing_low": fib.get("swing_low"),
    }

def build_trade_summary(out: dict) -> str:
    tkr = out.get("ticker", "?").replace("=X", "")
    s = out.get("summary", {})
    tf = out.get("timeframes", {})
    rk = out.get("risk", {})
    pz = out.get("position", {})
    conf = out.get("final_confidence", None)

    action = s.get("action", "HOLD")
    wscore = s.get("weighted_score", 0.0)

    # Determinar Bias
    score_4h = tf.get("4H", {}).get("score", 0.0)
    score_1h = tf.get("1H", {}).get("score", 0.0)
    score_30m = tf.get("30m", {}).get("score", 0.0)

    if wscore <= -0.35:
        bias_txt = "Bajista claro"
    elif wscore <= -0.15:
        bias_txt = "Bajista moderado"
    elif wscore >= 0.35:
        bias_txt = "Alcista claro"
    elif wscore >= 0.15:
        bias_txt = "Alcista moderado"
    else:
        bias_txt = "Neutral / Sin dirección clara"

    # Determinar acción con emoji
    if action == "BUY":
        action_txt = "COMPRAR (BUY)"
        action_emoji = "🟢"
    elif action == "SELL":
        action_txt = "VENDER (SELL)"
        action_emoji = "🔴"
    else:
        action_txt = "ESPERAR (HOLD)"
        action_emoji = "🟡"

    # Confianza con descripción
    conf_val = conf if isinstance(conf, (int, float)) else 0.0
    if conf_val >= 0.75:
        conf_desc = "alta"
    elif conf_val >= 0.50:
        conf_desc = "buena, no máxima"
    elif conf_val >= 0.30:
        conf_desc = "moderada"
    else:
        conf_desc = "baja"

    # Contexto de timeframes
    def tf_estado(score: float) -> str:
        if score >= 0.45:
            return "alcista fuerte"
        elif score >= 0.20:
            return "alcista"
        elif score <= -0.45:
            return "bajista fuerte"
        elif score <= -0.20:
            return "bajista"
        else:
            return "plano"

    ctx_4h = "manda fuerte" if abs(score_4h) >= 0.45 else ("acompaña" if abs(score_4h) >= 0.20 else "neutral")
    ctx_1h = "acompaña" if (score_4h * score_1h > 0 and abs(score_1h) >= 0.20) else ("diverge" if score_4h * score_1h < 0 else "neutral")
    ctx_30m = "está plano" if abs(score_30m) < 0.25 else ("alineado" if score_4h * score_30m > 0 else "diverge")

    impulso_warning = " → no entrar por impulso" if abs(score_30m) < 0.25 else ""
    contexto_txt = f"4H {ctx_4h}; 1H {ctx_1h}; 30m {ctx_30m}{impulso_warning}"

    # Detectar divergencias en los timeframes
    divergences_found = []
    for tf_name in ["4H", "1H", "30m"]:
        div = tf.get(tf_name, {}).get("divergence", {})
        if div and div.get("type"):
            divergences_found.append(f"{tf_name}: {div.get('description', '')}")

    # Header
    lines = [
        f"{'─' * 50}",
        f"  📊 SEÑAL DE TRADING: {tkr}",
        f"{'─' * 50}",
        "",
        f"📉 Bias: {bias_txt}",
        f"{action_emoji} Acción: {action_txt}",
        f"🎯 Confianza: {conf_val:.2f} ({conf_desc})",
        f"🔍 Contexto: {contexto_txt}",
    ]

    # Mostrar divergencias si existen
    if divergences_found:
        lines.extend([
            "",
            f"{'─' * 50}",
            "🔀 Divergencias RSI detectadas",
            f"{'─' * 50}",
        ])
        for div_txt in divergences_found:
            lines.append(f"  • {div_txt}")

    # Plan de trade
    if action != "HOLD" and rk.get("sl") is not None and rk.get("tp") is not None:
        entry = rk.get("entry", 0)
        sl = rk.get("sl", 0)
        tp = rk.get("tp", 0)
        sl_tag = rk.get("sl_tag", "ATR")
        tp_tag = rk.get("tp_tag", "ATR")
        
        # Calcular R:R real
        sl_dist = abs(entry - sl)
        tp_dist = abs(entry - tp)
        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0

        lines.extend([
            "",
            f"{'─' * 50}",
            "✅ Plan de trade",
            f"{'─' * 50}",
            f"  • Entrada: {entry:.5f}",
            f"  • Stop Loss: {sl:.5f} ({sl_tag})",
            f"  • Take Profit: {tp:.5f} ({tp_tag})",
            f"  • R:R: ~1:{rr_ratio:.1f}",
        ])

        # Lotaje
        if pz and pz.get("lots") is not None:
            lots = pz.get("lots", 0)
            risk_usd = pz.get("risk_usd_actual", 0)
            risk_pct = pz.get("risk_pct", 0.01) * 100
            lines.append(f"  • Tamaño: {lots:.2f} lotes (riesgo ≈ ${risk_usd:.2f}, ~{risk_pct:.0f}%)")

    # Timing recomendado
    lines.extend([
        "",
        f"{'─' * 50}",
        "⏱️ Timing recomendado",
        f"{'─' * 50}",
    ])

    if abs(score_30m) < 0.25:
        lines.extend([
            "  • Ideal: esperar ruptura o retesteo en 30m y entrar en confirmación.",
            "  • Aceptable: entrada ya solo si respetás el SL sin moverlo.",
        ])
    else:
        lines.extend([
            "  • Condiciones alineadas: entrada válida en próxima vela de confirmación.",
            "  • Alternativa: esperar pullback a zona de valor para mejor precio.",
        ])

    # Trailing Stop (si está disponible)
    trailing = out.get("trailing_stop", {})
    if trailing and action != "HOLD":
        entry = rk.get("entry", 0)
        decimals = 3 if "JPY" in tkr else 5
        
        lines.extend([
            "",
            f"{'─' * 50}",
            "📈 Trailing Stop sugerido",
            f"{'─' * 50}",
            f"  • Al alcanzar +1R ({trailing.get('breakeven_trigger', 0):.{decimals}f}):",
            f"    → Mover SL a breakeven ({trailing.get('breakeven_sl', 0):.{decimals}f})",
            f"  • Al alcanzar +1.5R ({trailing.get('trail_1_trigger', 0):.{decimals}f}):",
            f"    → Mover SL a +0.5R ({trailing.get('trail_1_sl', 0):.{decimals}f})",
            f"  • Al alcanzar +2R ({trailing.get('trail_2_trigger', 0):.{decimals}f}):",
            f"    → Mover SL a +1R ({trailing.get('trail_2_sl', 0):.{decimals}f})",
        ])

    # Invalidación
    lines.extend([
        "",
        f"{'─' * 50}",
        "⚠️ Invalidación",
        f"{'─' * 50}",
    ])

    if action == "BUY":
        lines.append("  • Si el precio rompe el SL con decisión → salir sin dudar.")
    elif action == "SELL":
        lines.append("  • Si el precio rompe el SL con decisión → salir sin dudar.")
    else:
        lines.append("  • Esperar señal clara antes de operar.")

    lines.append(f"{'─' * 50}")

    return "\n".join(lines)

def get_signal(
    ticker: str = "EURUSD=X",
    rr: float = 2.0,
    atr_mult_sl: float = 1.5,
    use_fibonacci: bool = True,
    fib_timeframe: str = "4H",
    fib_bars: int = 140,
    balance_usd: float = 1184.0,   # <- TU BALANCE
    risk_pct: float = 0.01         # <- 1% riesgo por trade
) -> Dict[str, object]:
    tf_results = {}
    tf_dfs = {}

    for tf, cfg in TF_MAP.items():
        df = fetch_ohlc(ticker, cfg.interval, cfg.lookback)
        if len(df) < 120:
            raise ValueError(f"Datos insuficientes en {tf} ({cfg.interval}). Filas: {len(df)}")
        tf_dfs[tf] = df
        tf_results[tf] = score_timeframe(df)

    summary = aggregate_signal(tf_results)

    entry = tf_results["30m"]["close"]
    atr_for_risk = tf_results["1H"]["atr"]

    risk_atr = compute_sl_tp_atr(
        action=summary["action"],
        entry=entry,
        atr_value=atr_for_risk,
        rr=rr,
        atr_mult_sl=atr_mult_sl
    )

    fib_info = None
    risk_final = risk_atr

    if use_fibonacci and summary["action"] != "HOLD":
        fib_tf = fib_timeframe if fib_timeframe in tf_dfs else "4H"
        fib_info = compute_fib_levels(tf_dfs[fib_tf], bars=fib_bars)
        risk_final = adjust_sl_tp_with_fib(
            action=summary["action"],
            entry=entry,
            atr_sl=risk_atr["sl"],
            atr_tp=risk_atr["tp"],
            fib=fib_info,
            max_adjust_pct_of_atr=0.75
        )
        risk_final["atr_details"] = risk_atr
        risk_final["fib_timeframe"] = fib_tf

    final_conf = compute_final_confidence(summary["weighted_score"], summary["agreement"], tf_results)

    # Position sizing (lotaje)
    position = None
    if summary["action"] != "HOLD" and risk_final.get("sl") is not None:
        position = position_size_lots(
            ticker=ticker,
            balance_usd=balance_usd,
            risk_pct=risk_pct,
            entry=risk_final["entry"],
            sl=risk_final["sl"],
            min_lot=0.01,
            lot_step=0.01
        )

    return {
        "ticker": ticker,
        "timeframes": tf_results,
        "summary": summary,
        "final_confidence": final_conf,
        "risk": risk_final,
        "position": position,
        "fib_used": fib_info if use_fibonacci else None,
        "settings": {
            "rr": rr,
            "atr_mult_sl": atr_mult_sl,
            "use_fibonacci": use_fibonacci,
            "fib_timeframe": fib_timeframe,
            "fib_bars": fib_bars,
            "balance_usd": balance_usd,
            "risk_pct": risk_pct
        }
    }


# ----------------------------
# Run
# ----------------------------

# Lista de pares Forex populares para escanear
FOREX_UNIVERSE = [
    # Majors
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "AUDUSD=X",
    "USDCAD=X",
    "NZDUSD=X",
    # Crosses
    "EURGBP=X",
    "EURJPY=X",
    "GBPJPY=X",
    "AUDJPY=X",
    "EURAUD=X",
    "GBPAUD=X",
    "EURCHF=X",
    "GBPCHF=X",
    "AUDNZD=X",
    "CADJPY=X",
    "NZDJPY=X",
]

# ----------------------------
# Correlaciones conocidas entre pares
# ----------------------------
# Pares que tienden a moverse JUNTOS (correlación positiva alta)
POSITIVE_CORRELATIONS = [
    ("EURUSD", "GBPUSD"),      # Ambos vs USD
    ("AUDUSD", "NZDUSD"),      # Commodities vs USD
    ("EURJPY", "GBPJPY"),      # Ambos vs JPY
    ("EURCHF", "EURGBP"),      # Ambos base EUR
    ("AUDJPY", "NZDJPY"),      # Commodities vs JPY
]

# Pares que tienden a moverse OPUESTO (correlación negativa alta)
# Si EURUSD sube y USDCHF baja = MISMO trade (ambos short USD)
NEGATIVE_CORRELATIONS = [
    ("EURUSD", "USDCHF"),      # EUR vs CHF contra USD
    ("GBPUSD", "USDCHF"),      # GBP vs CHF contra USD
    ("AUDUSD", "USDCAD"),      # Commodities opuestas
    ("EURUSD", "USDCAD"),      # EUR vs CAD contra USD
]


def calculate_rr_ratio(entry: float, sl: float, tp: float) -> float:
    """Calcula el ratio Risk:Reward real."""
    if not all([entry, sl, tp]):
        return 0.0
    sl_dist = abs(entry - sl)
    tp_dist = abs(entry - tp)
    return tp_dist / sl_dist if sl_dist > 0 else 0.0


def get_trading_session() -> Dict[str, object]:
    """
    Determina la sesión de trading actual basada en hora UTC.
    Retorna info sobre qué sesión está activa y cuáles pares son óptimos.
    """
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    
    sessions = {
        "sydney": {"start": 21, "end": 6, "pairs": ["AUDUSD", "NZDUSD", "AUDJPY", "NZDJPY", "AUDNZD"]},
        "tokyo": {"start": 23, "end": 8, "pairs": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY"]},
        "london": {"start": 8, "end": 17, "pairs": ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "GBPCHF"]},
        "newyork": {"start": 13, "end": 22, "pairs": ["EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"]},
    }
    
    active_sessions = []
    optimal_pairs = set()
    
    for session_name, info in sessions.items():
        start, end = info["start"], info["end"]
        # Manejar sesiones que cruzan medianoche
        if start > end:  # Sydney: 22-7
            is_active = hour >= start or hour < end
        else:
            is_active = start <= hour < end
        
        if is_active:
            active_sessions.append(session_name)
            optimal_pairs.update(info["pairs"])
    
    # Determinar volatilidad esperada
    if "london" in active_sessions and "newyork" in active_sessions:
        volatility = "ALTA"
        overlap = "London-NY (mejor momento)"
    elif "tokyo" in active_sessions and "london" in active_sessions:
        volatility = "MEDIA-ALTA"
        overlap = "Tokyo-London"
    elif len(active_sessions) == 0:
        volatility = "BAJA"
        overlap = "Entre sesiones"
        active_sessions = ["ninguna"]
    else:
        volatility = "MEDIA"
        overlap = None
    
    return {
        "utc_time": now_utc.strftime("%H:%M UTC"),
        "active_sessions": active_sessions,
        "optimal_pairs": list(optimal_pairs),
        "volatility": volatility,
        "overlap": overlap
    }


def check_correlation_conflicts(results: List[dict]) -> List[Dict[str, object]]:
    """
    Detecta conflictos de correlación entre señales activas.
    Retorna lista de alertas sobre trades potencialmente duplicados.
    """
    conflicts = []
    
    # Extraer pares con señales activas (no HOLD)
    active_signals = {}
    for r in results:
        action = r["summary"]["action"]
        if action != "HOLD":
            pair = r["ticker"].replace("=X", "")
            active_signals[pair] = action
    
    # Verificar correlaciones positivas (mismo movimiento = trade duplicado si misma dirección)
    for pair1, pair2 in POSITIVE_CORRELATIONS:
        if pair1 in active_signals and pair2 in active_signals:
            action1, action2 = active_signals[pair1], active_signals[pair2]
            if action1 == action2:
                conflicts.append({
                    "type": "DUPLICADO",
                    "pairs": (pair1, pair2),
                    "actions": (action1, action2),
                    "warning": f"⚠️ {pair1} y {pair2} están correlacionados positivamente. "
                               f"Ambos {action1} = doble exposición al mismo movimiento.",
                    "suggestion": "Elegir solo uno o reducir tamaño en ambos."
                })
    
    # Verificar correlaciones negativas (movimiento opuesto = mismo trade si direcciones opuestas)
    for pair1, pair2 in NEGATIVE_CORRELATIONS:
        if pair1 in active_signals and pair2 in active_signals:
            action1, action2 = active_signals[pair1], active_signals[pair2]
            # Si uno es BUY y otro SELL en pares negativamente correlacionados = mismo trade
            if (action1 == "BUY" and action2 == "SELL") or (action1 == "SELL" and action2 == "BUY"):
                conflicts.append({
                    "type": "DUPLICADO",
                    "pairs": (pair1, pair2),
                    "actions": (action1, action2),
                    "warning": f"⚠️ {pair1} ({action1}) y {pair2} ({action2}) están correlacionados negativamente. "
                               f"Esto equivale al mismo trade (doble exposición).",
                    "suggestion": "Elegir solo uno o reducir tamaño en ambos."
                })
    
    return conflicts


def calculate_trailing_stop(entry: float, sl: float, tp: float, action: str) -> Dict[str, float]:
    """
    Calcula niveles sugeridos para trailing stop.
    Basado en: mover SL a breakeven cuando alcance 1R, luego trail.
    """
    if not all([entry, sl, tp]):
        return {}
    
    sl_dist = abs(entry - sl)
    
    # Niveles de trailing
    if action == "BUY":
        breakeven_trigger = entry + sl_dist  # Cuando ganemos 1R
        trail_1 = entry + (sl_dist * 1.5)     # Mover SL a +0.5R
        trail_2 = entry + (sl_dist * 2.0)     # Mover SL a +1R
    elif action == "SELL":
        breakeven_trigger = entry - sl_dist
        trail_1 = entry - (sl_dist * 1.5)
        trail_2 = entry - (sl_dist * 2.0)
    else:
        return {}
    
    return {
        "breakeven_trigger": breakeven_trigger,
        "breakeven_sl": entry,  # SL se mueve a entry
        "trail_1_trigger": trail_1,
        "trail_1_sl": entry + (sl_dist * 0.5) if action == "BUY" else entry - (sl_dist * 0.5),
        "trail_2_trigger": trail_2,
        "trail_2_sl": entry + sl_dist if action == "BUY" else entry - sl_dist,
    }


def calculate_multi_tp(
    entry: float, 
    sl: float, 
    tp: float, 
    action: str,
    total_lots: float,
    ticker: str = ""
) -> Dict[str, object]:
    """
    Calcula Take Profits Parciales para gestionar la posición de forma escalonada.
    
    Estrategia:
    - TP1 (50% de lots): Al alcanzar 1R → Asegurar ganancia, mover SL a breakeven
    - TP2 (30% de lots): Al alcanzar 1.5R → Asegurar más, trail SL
    - TP3 (20% de lots): Dejar correr hasta TP original o trailing
    
    Retorna:
    - tp1, tp2, tp3: Niveles de precio
    - lots1, lots2, lots3: Lotes para cada nivel
    - total_if_all_hit: Ganancia total si todos los TPs se alcanzan
    - description: Resumen del plan
    """
    if not all([entry, sl, tp, total_lots]):
        return {}
    
    sl_dist = abs(entry - sl)
    tp_dist = abs(entry - tp)
    decimals = 3 if "JPY" in ticker.upper() else 5
    
    # Distribución de lotes
    lots1 = round(total_lots * 0.50, 2)  # 50%
    lots2 = round(total_lots * 0.30, 2)  # 30%
    lots3 = round(total_lots - lots1 - lots2, 2)  # 20% restante
    
    # Calcular niveles de TP
    if action == "BUY":
        tp1 = entry + sl_dist          # 1R
        tp2 = entry + (sl_dist * 1.5)  # 1.5R
        tp3 = tp                        # TP original (target completo)
    elif action == "SELL":
        tp1 = entry - sl_dist
        tp2 = entry - (sl_dist * 1.5)
        tp3 = tp
    else:
        return {}
    
    # Calcular pips de cada TP
    pip = 0.01 if "JPY" in ticker.upper() else 0.0001
    tp1_pips = abs(tp1 - entry) / pip
    tp2_pips = abs(tp2 - entry) / pip
    tp3_pips = abs(tp3 - entry) / pip
    sl_pips = sl_dist / pip
    
    # Calcular R de cada nivel
    tp1_r = 1.0
    tp2_r = 1.5
    tp3_r = tp_dist / sl_dist if sl_dist > 0 else 0
    
    # Calcular ganancia ponderada si todos los TPs se alcanzan
    # (en R totales: contribución de cada parte)
    weighted_r = (0.50 * tp1_r) + (0.30 * tp2_r) + (0.20 * tp3_r)
    
    description_lines = [
        f"📊 PLAN DE SALIDAS PARCIALES:",
        f"   ├─ TP1 (50%): {tp1:.{decimals}f} → {lots1:.2f} lots @ +{tp1_pips:.0f} pips (1R)",
        f"   ├─ TP2 (30%): {tp2:.{decimals}f} → {lots2:.2f} lots @ +{tp2_pips:.0f} pips (1.5R)",
        f"   └─ TP3 (20%): {tp3:.{decimals}f} → {lots3:.2f} lots @ +{tp3_pips:.0f} pips ({tp3_r:.1f}R)",
        f"   💰 R ponderado total: {weighted_r:.2f}R (si todos los TPs se alcanzan)"
    ]
    
    return {
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp1_pips": round(tp1_pips, 0),
        "tp2_pips": round(tp2_pips, 0),
        "tp3_pips": round(tp3_pips, 0),
        "lots1": lots1,
        "lots2": lots2,
        "lots3": lots3,
        "tp1_r": tp1_r,
        "tp2_r": tp2_r,
        "tp3_r": round(tp3_r, 2),
        "weighted_r": round(weighted_r, 2),
        "description": "\n".join(description_lines)
    }


def export_signals_to_csv(results: List[dict], filename: str = None) -> str:
    """
    Exporta las señales a un archivo CSV para tracking histórico.
    """
    if not results:
        return "No hay señales para exportar."
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"signals_{timestamp}.csv"
    
    # Preparar datos
    rows = []
    for r in results:
        tkr = r["ticker"].replace("=X", "")
        action = r["summary"]["action"]
        conf = r["final_confidence"]
        wscore = r["summary"]["weighted_score"]
        
        rk = r.get("risk", {})
        pz = r.get("position", {})
        
        entry = rk.get("entry", 0)
        sl = rk.get("sl", 0)
        tp = rk.get("tp", 0)
        
        rr_ratio = calculate_rr_ratio(entry, sl, tp)
        lots = pz.get("lots", 0) if pz else 0
        risk_usd = pz.get("risk_usd_actual", 0) if pz else 0
        
        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pair": tkr,
            "action": action,
            "confidence": round(conf, 2),
            "weighted_score": round(wscore, 2),
            "entry": round(entry, 5) if entry else "",
            "stop_loss": round(sl, 5) if sl else "",
            "take_profit": round(tp, 5) if tp else "",
            "rr_ratio": round(rr_ratio, 2),
            "lots": round(lots, 2),
            "risk_usd": round(risk_usd, 2),
            "sl_tag": rk.get("sl_tag", ""),
            "tp_tag": rk.get("tp_tag", ""),
        })
    
    # Escribir CSV
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    return filepath


def build_session_info() -> str:
    """Genera información sobre la sesión de trading actual."""
    session = get_trading_session()
    
    lines = [
        f"{'═' * 60}",
        f"  ⏰ SESIÓN DE TRADING ACTUAL",
        f"{'═' * 60}",
        f"  Hora: {session['utc_time']}",
        f"  Sesiones activas: {', '.join(s.capitalize() for s in session['active_sessions'])}",
        f"  Volatilidad esperada: {session['volatility']}",
    ]
    
    if session['overlap']:
        lines.append(f"  💡 {session['overlap']}")
    
    lines.append(f"  Pares óptimos: {', '.join(session['optimal_pairs'][:8])}...")
    lines.append(f"{'═' * 60}")
    
    return "\n".join(lines)


def build_correlation_warnings(conflicts: List[dict]) -> str:
    """Genera advertencias de correlación."""
    if not conflicts:
        return ""
    
    lines = [
        f"\n{'═' * 60}",
        f"  ⚠️ ALERTAS DE CORRELACIÓN",
        f"{'═' * 60}",
    ]
    
    for c in conflicts:
        lines.append(f"\n  {c['warning']}")
        lines.append(f"  💡 Sugerencia: {c['suggestion']}")
    
    lines.append(f"\n{'═' * 60}")
    
    return "\n".join(lines)


def scan_market(
    tickers: List[str] = None,
    balance_usd: float = 1184.0,
    risk_pct: float = 0.01,
    rr: float = 2.0,
    atr_mult_sl: float = 1.5,
    min_confidence: float = 0.0,
    min_rr: float = 1.0,
    show_hold: bool = False,
    filter_by_session: bool = False,
    filter_adr_exhausted: bool = True,
    adr_warning_threshold: float = 70.0
) -> List[dict]:
    """
    Escanea múltiples pares y retorna lista ordenada por confianza.
    
    Parámetros:
    - min_confidence: Confianza mínima para incluir señal
    - min_rr: Ratio R:R mínimo para incluir señal
    - filter_by_session: Si True, prioriza pares óptimos para la sesión actual
    - filter_adr_exhausted: Si True, excluye pares que ya movieron >70% del ADR
    - adr_warning_threshold: % del ADR para considerar movimiento agotado
    """
    if tickers is None:
        tickers = FOREX_UNIVERSE
    
    # Obtener info de sesión si se requiere filtrar
    session_info = get_trading_session() if filter_by_session else None
    optimal_pairs = session_info["optimal_pairs"] if session_info else []
    
    results = []
    
    print(f"\n{'═' * 60}")
    print(f"  🔍 ESCANEANDO {len(tickers)} PARES DE FOREX...")
    print(f"{'═' * 60}\n")
    
    for i, ticker in enumerate(tickers, 1):
        symbol = ticker.replace("=X", "")
        try:
            print(f"  [{i:2d}/{len(tickers)}] Analizando {symbol}...", end=" ", flush=True)
            
            # Calcular ADR primero (filtro rápido)
            adr_info = calculate_adr(ticker, period=14)
            
            signal = get_signal(
                ticker=ticker,
                rr=rr,
                atr_mult_sl=atr_mult_sl,
                use_fibonacci=True,
                fib_timeframe="4H",
                fib_bars=140,
                balance_usd=balance_usd,
                risk_pct=risk_pct
            )
            
            action = signal["summary"]["action"]
            conf = signal["final_confidence"]
            
            # Filtrar HOLD
            if action == "HOLD" and not show_hold:
                print(f"⏸️  HOLD (skip)")
                continue
            
            # Filtrar por confianza
            if conf < min_confidence:
                print(f"⚠️  Conf {conf:.2f} < {min_confidence:.2f} (skip)")
                continue
            
            # Filtrar por ADR agotado
            if filter_adr_exhausted and adr_info.get("exhausted", False):
                adr_pct = adr_info.get("today_pct", 0)
                print(f"🔋 ADR {adr_pct:.0f}% agotado (skip)")
                continue
            
            # Calcular y filtrar por R:R
            rk = signal.get("risk", {})
            pz = signal.get("position", {})
            entry = rk.get("entry", 0)
            sl = rk.get("sl", 0)
            tp = rk.get("tp", 0)
            lots = pz.get("lots", 0) if pz else 0
            actual_rr = calculate_rr_ratio(entry, sl, tp)
            
            if actual_rr < min_rr:
                print(f"📉 R:R {actual_rr:.1f} < {min_rr:.1f} (skip)")
                continue
            
            # Ajustar SL/TP por spread
            spread_adj = adjust_sl_tp_for_spread(entry, sl, tp, action, ticker, buffer_multiplier=1.0)
            signal["spread_adjustment"] = spread_adj
            
            # Agregar trailing stop info al signal
            if action != "HOLD":
                signal["trailing_stop"] = calculate_trailing_stop(entry, sl, tp, action)
                # Agregar Multi-TP (take profits parciales) - usar TP ajustado por spread
                tp_adj = spread_adj.get("tp_adjusted", tp)
                signal["multi_tp"] = calculate_multi_tp(entry, sl, tp_adj, action, lots, ticker)
                # Detectar zonas de liquidez
                signal["liquidity"] = detect_liquidity_zones(ticker, timeframe="4h")
            
            # Agregar info de ADR al signal
            signal["adr"] = adr_info
            
            # Marcar si es par óptimo para la sesión
            signal["session_optimal"] = symbol in optimal_pairs if filter_by_session else True
            
            # Calcular score de calidad A/B/C
            signal["quality"] = calculate_quality_score(signal)
            
            results.append(signal)
            
            emoji = "🟢" if action == "BUY" else ("🔴" if action == "SELL" else "🟡")
            session_mark = "⭐" if signal.get("session_optimal") else ""
            adr_pct = adr_info.get("today_pct", 0)
            adr_mark = f"ADR:{adr_pct:.0f}%" if adr_pct > 0 else ""
            quality = signal.get("quality", {})
            grade = quality.get("grade", "?")
            print(f"{emoji} {action} | Conf: {conf:.2f} | R:R: 1:{actual_rr:.1f} | {adr_mark} | Grade: {grade} {session_mark}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:40]}")
            continue
    
    # Ordenar por confianza (mayor primero)
    results.sort(key=lambda x: x["final_confidence"], reverse=True)
    
    return results


def build_ranking_summary(results: List[dict], top_n: int = 10) -> str:
    """
    Genera un resumen compacto del ranking de oportunidades.
    """
    if not results:
        return "\n⚠️ No se encontraron oportunidades con los criterios actuales.\n"
    
    top = results[:top_n]
    
    lines = [
        "",
        f"{'═' * 90}",
        f"  🏆 TOP {len(top)} OPORTUNIDADES (ordenadas por confianza)",
        f"{'═' * 90}",
        "",
        f"  {'#':<3} {'Par':<10} {'Acción':<8} {'Conf':<6} {'Grade':<7} {'Bias':<14} {'Entry':<10} {'SL':<10} {'TP':<10} {'Lots':<6}",
        f"  {'-' * 87}",
    ]
    
    for i, sig in enumerate(top, 1):
        tkr = sig["ticker"].replace("=X", "")
        action = sig["summary"]["action"]
        conf = sig["final_confidence"]
        wscore = sig["summary"]["weighted_score"]
        
        # Quality grade
        quality = sig.get("quality", {})
        grade = quality.get("grade", "?")
        score = quality.get("score", 0)
        grade_str = f"{grade}({score:.0f})"
        
        # Bias
        if wscore <= -0.35:
            bias = "Bajista claro"
        elif wscore <= -0.15:
            bias = "Bajista mod."
        elif wscore >= 0.35:
            bias = "Alcista claro"
        elif wscore >= 0.15:
            bias = "Alcista mod."
        else:
            bias = "Neutral"
        
        # Risk data
        rk = sig.get("risk", {})
        pz = sig.get("position", {})
        entry = rk.get("entry", 0)
        sl = rk.get("sl", 0)
        tp = rk.get("tp", 0)
        lots = pz.get("lots", 0) if pz else 0
        
        # Emoji
        emoji = "🟢" if action == "BUY" else ("🔴" if action == "SELL" else "🟡")
        
        # Formatear precios según el par
        decimals = 3 if "JPY" in tkr else 5
        entry_str = f"{entry:.{decimals}f}" if entry else "-"
        sl_str = f"{sl:.{decimals}f}" if sl else "-"
        tp_str = f"{tp:.{decimals}f}" if tp else "-"
        lots_str = f"{lots:.2f}" if lots else "-"
        
        lines.append(
            f"  {i:<3} {tkr:<10} {emoji}{action:<7} {conf:<6.2f} {grade_str:<7} {bias:<14} {entry_str:<10} {sl_str:<10} {tp_str:<10} {lots_str:<6}"
        )
    
    lines.extend([
        f"  {'-' * 77}",
        "",
        f"  📊 Total oportunidades encontradas: {len(results)}",
        f"  💡 Mostrando top {len(top)} ordenadas por confianza",
        f"  🅰️ = Alta probabilidad (≥80) | 🅱️ = Aceptable (≥60) | ©️ = Baja calidad (<60)",
        "",
    ])
    
    # Resumen de distribución
    buys = sum(1 for r in results if r["summary"]["action"] == "BUY")
    sells = sum(1 for r in results if r["summary"]["action"] == "SELL")
    
    lines.extend([
        f"  📈 Señales BUY: {buys}  |  📉 Señales SELL: {sells}",
        f"{'═' * 70}",
    ])
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SISTEMA DE EJECUCIÓN EN VIVO
# ═══════════════════════════════════════════════════════════════

import time
import winsound  # Windows only - para alertas sonoras


def is_forex_market_open() -> Dict[str, object]:
    """
    Verifica si el mercado Forex está abierto.
    
    Forex opera 24/5:
    - Abre: Domingo 5:00 PM EST (22:00 UTC)
    - Cierra: Viernes 5:00 PM EST (22:00 UTC)
    
    Retorna: {"is_open": bool, "reason": str, "next_open": str}
    """
    from datetime import timezone, timedelta
    
    # Obtener hora UTC actual
    now_utc = datetime.now(timezone.utc)
    weekday = now_utc.weekday()  # 0=Lunes, 6=Domingo
    hour_utc = now_utc.hour
    
    # Forex cierra Viernes 22:00 UTC y abre Domingo 22:00 UTC
    # Sábado = cerrado todo el día
    # Domingo antes de 22:00 UTC = cerrado
    # Viernes después de 22:00 UTC = cerrado
    
    if weekday == 5:  # Sábado
        return {
            "is_open": False,
            "reason": "🔴 Mercado CERRADO - Sábado",
            "next_open": "Domingo 22:00 UTC"
        }
    elif weekday == 6 and hour_utc < 22:  # Domingo antes de apertura
        return {
            "is_open": False,
            "reason": "🔴 Mercado CERRADO - Domingo (antes de apertura)",
            "next_open": "Domingo 22:00 UTC"
        }
    elif weekday == 4 and hour_utc >= 22:  # Viernes después de cierre
        return {
            "is_open": False,
            "reason": "🔴 Mercado CERRADO - Fin de semana",
            "next_open": "Domingo 22:00 UTC"
        }
    else:
        session = get_trading_session()
        active = ", ".join(session['active_sessions']).title()
        return {
            "is_open": True,
            "reason": f"🟢 Mercado ABIERTO - Sesión: {active}",
            "next_open": None
        }


def play_alert_sound(alert_type: str = "signal"):
    """
    Reproduce un sonido de alerta (Windows).
    """
    try:
        if alert_type == "signal":
            # Beep de alta frecuencia para señal
            winsound.Beep(1000, 500)  # 1000 Hz por 500ms
            time.sleep(0.1)
            winsound.Beep(1200, 500)
        elif alert_type == "grade_a":
            # Triple beep para grado A
            for _ in range(3):
                winsound.Beep(1500, 300)
                time.sleep(0.1)
        elif alert_type == "error":
            winsound.Beep(400, 1000)  # Tono bajo para error
    except Exception:
        pass  # Si falla el sonido, continuar


def clear_screen():
    """Limpia la pantalla de la consola."""
    os.system('cls' if os.name == 'nt' else 'clear')


def run_single_scan(
    balance_usd: float,
    risk_pct: float,
    min_confidence: float,
    min_rr: float,
    top_n: int,
    show_details: int,
    filter_by_session: bool,
    filter_adr: bool,
    export_csv: bool,
    alert_on_grade_a: bool
) -> List[dict]:
    """
    Ejecuta un escaneo completo del mercado.
    """
    # Info de sesión
    print(build_session_info())
    
    # Mostrar trades abiertos (si hay)
    try:
        from trades import list_open_trades, load_trades, save_last_signals
        trades_data = load_trades()
        if trades_data.get("open"):
            list_open_trades(show_alerts=True)
    except ImportError:
        pass
    
    # Escaneo
    results = scan_market(
        tickers=FOREX_UNIVERSE,
        balance_usd=balance_usd,
        risk_pct=risk_pct,
        min_confidence=min_confidence,
        min_rr=min_rr,
        show_hold=False,
        filter_by_session=filter_by_session,
        filter_adr_exhausted=filter_adr
    )
    
    # Guardar señales para el sistema de tracking
    try:
        from trades import save_last_signals
        # Preparar señales serializables
        signals_to_save = []
        for r in results:
            sig = {
                "ticker": r.get("ticker"),
                "summary": r.get("summary"),
                "risk": r.get("risk"),
                "position": r.get("position"),
                "quality": r.get("quality"),
                "spread_adjustment": r.get("spread_adjustment"),
                "final_confidence": r.get("final_confidence"),
            }
            signals_to_save.append(sig)
        save_last_signals(signals_to_save)
    except ImportError:
        pass
    
    # Ranking
    print(build_ranking_summary(results, top_n=top_n))
    
    # Alertas de correlación
    conflicts = check_correlation_conflicts(results)
    if conflicts:
        print(build_correlation_warnings(conflicts))
    
    # Detalle de mejores señales
    if results and show_details > 0:
        print(f"\n{'═' * 60}")
        print(f"  📋 DETALLE DE LAS TOP {min(show_details, len(results))} SEÑALES")
        print(f"{'═' * 60}")
        
        for sig in results[:show_details]:
            print(build_trade_summary(sig))
    
    # Exportar a CSV
    if export_csv and results:
        csv_path = export_signals_to_csv(results)
        print(f"\n📁 Señales exportadas a: {csv_path}")
    
    # Alertas de grado A
    if alert_on_grade_a:
        grade_a_signals = [r for r in results if r.get("quality", {}).get("grade") == "A"]
        if grade_a_signals:
            print(f"\n🔔 ¡ALERTA! {len(grade_a_signals)} señal(es) GRADO A detectada(s):")
            for sig in grade_a_signals:
                tkr = sig["ticker"].replace("=X", "")
                action = sig["summary"]["action"]
                score = sig.get("quality", {}).get("score", 0)
                print(f"   🅰️ {tkr} - {action} (Score: {score})")
            play_alert_sound("grade_a")
    
    # Recordatorio de comandos de trades
    print(f"\n{'─' * 50}")
    print(f"  💡 Para abrir trade: python trades.py --open N")
    print(f"  💡 Para ver trades:  python trades.py --list")
    print(f"{'─' * 50}")
    
    return results


def run_scanner(
    balance_usd: float = 1184.0,
    risk_pct: float = 0.01,
    min_confidence: float = 0.50,
    min_rr: float = 1.5,
    top_n: int = 10,
    show_details: int = 3,
    filter_by_session: bool = True,
    filter_adr: bool = True,
    export_csv: bool = True,
    live_mode: bool = False,
    update_interval: int = 15,
    alert_on_grade_a: bool = True,
    respect_market_hours: bool = True
):
    """
    Función principal que ejecuta el scanner.
    
    Modos:
    - live_mode=False: Ejecuta una vez y termina
    - live_mode=True: Loop continuo cada update_interval minutos
    
    Parámetros:
    - update_interval: Minutos entre cada escaneo (si live_mode=True)
    - respect_market_hours: Si True, pausa cuando el mercado está cerrado
    """
    
    if not live_mode:
        # Modo simple: una ejecución
        run_single_scan(
            balance_usd, risk_pct, min_confidence, min_rr,
            top_n, show_details, filter_by_session, filter_adr,
            export_csv, alert_on_grade_a
        )
        return
    
    # ═══════════════════════════════════════════════════════════════
    # MODO LIVE - Loop continuo
    # ═══════════════════════════════════════════════════════════════
    scan_count = 0
    
    print(f"\n{'═' * 70}")
    print(f"  🔄 MODO LIVE ACTIVADO")
    print(f"  📊 Actualizando cada {update_interval} minutos")
    print(f"  ⏰ Presiona Ctrl+C para detener")
    print(f"{'═' * 70}\n")
    
    try:
        while True:
            scan_count += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Verificar si el mercado está abierto
            if respect_market_hours:
                market_status = is_forex_market_open()
                if not market_status["is_open"]:
                    print(f"\n[{current_time}] {market_status['reason']}")
                    print(f"  Próxima apertura: {market_status['next_open']}")
                    print(f"  Esperando 30 minutos para volver a verificar...")
                    time.sleep(30 * 60)  # Esperar 30 min y volver a verificar
                    continue
            
            # Limpiar pantalla para nuevo escaneo
            clear_screen()
            
            print(f"\n{'═' * 70}")
            print(f"  🔄 ESCANEO #{scan_count} - {current_time}")
            print(f"  ⏰ Próxima actualización en {update_interval} minutos")
            print(f"{'═' * 70}\n")
            
            # Ejecutar escaneo
            try:
                results = run_single_scan(
                    balance_usd, risk_pct, min_confidence, min_rr,
                    top_n, show_details, filter_by_session, filter_adr,
                    export_csv, alert_on_grade_a
                )
                
                if not results:
                    print("\n⚠️ No se encontraron señales en este escaneo.")
                
            except Exception as e:
                print(f"\n❌ Error en escaneo: {str(e)}")
                play_alert_sound("error")
            
            # Countdown hasta próximo escaneo
            print(f"\n{'─' * 50}")
            print(f"  ⏳ Próximo escaneo en {update_interval} minutos...")
            print(f"  (Presiona Ctrl+C para detener)")
            print(f"{'─' * 50}")
            
            # Esperar hasta próximo escaneo (con updates cada minuto)
            for remaining in range(update_interval, 0, -1):
                time.sleep(60)  # 1 minuto
                if remaining > 1:
                    print(f"  ⏳ {remaining - 1} minutos restantes...", end="\r")
    
    except KeyboardInterrupt:
        print(f"\n\n{'═' * 50}")
        print(f"  🛑 SCANNER DETENIDO")
        print(f"  Total de escaneos realizados: {scan_count}")
        print(f"{'═' * 50}\n")


def build_compact_signal(sig: dict) -> str:
    """
    Versión compacta de una señal individual para mostrar después del ranking.
    """
    tkr = sig["ticker"].replace("=X", "")
    action = sig["summary"]["action"]
    conf = sig["final_confidence"]
    wscore = sig["summary"]["weighted_score"]
    
    emoji = "🟢" if action == "BUY" else ("🔴" if action == "SELL" else "🟡")
    
    # Bias
    if wscore <= -0.35:
        bias = "Bajista claro"
    elif wscore <= -0.15:
        bias = "Bajista moderado"
    elif wscore >= 0.35:
        bias = "Alcista claro"
    elif wscore >= 0.15:
        bias = "Alcista moderado"
    else:
        bias = "Neutral"
    
    rk = sig.get("risk", {})
    pz = sig.get("position", {})
    
    entry = rk.get("entry", 0)
    sl = rk.get("sl", 0)
    tp = rk.get("tp", 0)
    sl_tag = rk.get("sl_tag", "ATR")
    tp_tag = rk.get("tp_tag", "ATR")
    lots = pz.get("lots", 0) if pz else 0
    risk_usd = pz.get("risk_usd_actual", 0) if pz else 0
    
    # Spread adjustment
    spread_adj = sig.get("spread_adjustment", {})
    sl_adj = spread_adj.get("sl_adjusted", sl)
    tp_adj = spread_adj.get("tp_adjusted", tp)
    spread_pips = spread_adj.get("spread_pips", 0)
    
    # R:R (usando valores originales para el cálculo)
    sl_dist = abs(entry - sl) if entry and sl else 0
    tp_dist = abs(entry - tp) if entry and tp else 0
    rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
    
    decimals = 3 if "JPY" in tkr else 5
    
    lines = [
        f"\n{'─' * 60}",
        f"  {emoji} {tkr} - {action}",
        f"{'─' * 60}",
        f"  Bias: {bias} | Confianza: {conf:.2f}",
        f"  Entry: {entry:.{decimals}f}",
        f"",
        f"  📍 NIVELES ORIGINALES (sin spread):",
        f"     SL: {sl:.{decimals}f} ({sl_tag}) | TP: {tp:.{decimals}f} ({tp_tag})",
        f"",
        f"  📍 NIVELES AJUSTADOS (+{spread_pips:.1f} pips spread):",
        f"     SL: {sl_adj:.{decimals}f} | TP: {tp_adj:.{decimals}f}  ← USAR ESTOS",
        f"",
        f"  R:R: ~1:{rr_ratio:.1f} | Lotes: {lots:.2f} (riesgo: ${risk_usd:.2f})",
    ]
    
    # Agregar info de ADR si está disponible
    adr_info = sig.get("adr", {})
    if adr_info:
        adr_pips = adr_info.get("adr_pips", 0)
        today_pct = adr_info.get("today_pct", 0)
        remaining = adr_info.get("remaining_pips", 0)
        adr_desc = adr_info.get("description", "")
        lines.append(f"  {adr_desc}")
    
    # Agregar info de Multi-TP si está disponible
    multi_tp = sig.get("multi_tp", {})
    if multi_tp and multi_tp.get("description"):
        lines.append("")
        lines.append(f"  {multi_tp['description']}")
    
    # Agregar info de Zonas de Liquidez si está disponible
    liquidity = sig.get("liquidity", {})
    if liquidity and liquidity.get("description"):
        lines.append("")
        lines.append(f"  {liquidity['description']}")
    
    # Agregar Score de Calidad
    quality = sig.get("quality", {})
    if quality and quality.get("description"):
        lines.append("")
        lines.append(f"  {quality['description']}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════
    # CONFIGURACIÓN
    # ═══════════════════════════════════════════════════════════════
    BALANCE_USD = 1184.0      # Tu balance en USD
    RISK_PCT = 0.05           # Riesgo por operación (1%)
    MIN_CONFIDENCE = 0.50     # Confianza mínima para mostrar (0.0 = todas)
    MIN_RR = 1.5              # Ratio R:R mínimo (1.5 = arriesgar 1 para ganar 1.5)
    TOP_N = 10                # Cuántas mostrar en el ranking
    SHOW_DETAILS = 0          # Mostrar detalle de las top N señales (0 = desactivado)
    FILTER_BY_SESSION = True  # Marcar pares óptimos para la sesión actual
    FILTER_ADR = True         # Filtrar pares con movimiento diario agotado
    EXPORT_CSV = True         # Exportar señales a CSV
    
    # ═══════════════════════════════════════════════════════════════
    # MODO DE EJECUCIÓN
    # ═══════════════════════════════════════════════════════════════
    LIVE_MODE = True          # True = loop continuo, False = una ejecución
    UPDATE_INTERVAL = 15       # Minutos entre actualizaciones (si LIVE_MODE=True)
    ALERT_ON_GRADE_A = True    # Notificar cuando hay señal grado A
    RESPECT_MARKET_HOURS = True # Solo ejecutar cuando Forex está abierto
    
    # ═══════════════════════════════════════════════════════════════
    # EJECUTAR
    # ═══════════════════════════════════════════════════════════════
    run_scanner(
        balance_usd=BALANCE_USD,
        risk_pct=RISK_PCT,
        min_confidence=MIN_CONFIDENCE,
        min_rr=MIN_RR,
        top_n=TOP_N,
        show_details=SHOW_DETAILS,
        filter_by_session=FILTER_BY_SESSION,
        filter_adr=FILTER_ADR,
        export_csv=EXPORT_CSV,
        live_mode=LIVE_MODE,
        update_interval=UPDATE_INTERVAL,
        alert_on_grade_a=ALERT_ON_GRADE_A,
        respect_market_hours=RESPECT_MARKET_HOURS
    )