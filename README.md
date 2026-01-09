# 📊 Forex Multi-Timeframe Signal Scanner

Sistema automatizado de análisis técnico para Forex con soporte multi-timeframe (4H, 1H, 30m).

## ✨ Características

- **Multi-timeframe analysis**: Combina señales de 4H (50%), 1H (30%) y 30m (20%)
- **18 pares de Forex**: Majors y crosses principales
- **Indicadores técnicos**: EMA(21,55), RSI(14), MACD, ATR(14)
- **Niveles Fibonacci**: Ajuste automático de SL/TP a niveles clave
- **Divergencias RSI**: Detección automática de divergencias alcistas/bajistas
- **Filtro ADR**: Evita entrar cuando el movimiento diario está agotado
- **Zonas de liquidez**: Detecta swing highs/lows donde se acumulan stops
- **Multi-TP**: Take profits parciales (50% @ 1R, 30% @ 1.5R, 20% runner)
- **Score de calidad A/B/C**: Clasifica cada setup por probabilidad
- **Ajuste de spread**: Compensa SL/TP por el spread típico del par
- **Correlación**: Alerta de doble exposición (ej: EURUSD SELL + USDCHF BUY)
- **Modo Live**: Loop continuo con actualización cada N minutos
- **Alertas sonoras**: Beep cuando hay señal Grado A

## 🚀 Instalación

```bash
pip install pandas yfinance
```

## ⚙️ Configuración

Edita las variables al final de `main.py`:

```python
BALANCE_USD = 1184.0      # Tu balance en USD
RISK_PCT = 0.01           # Riesgo por operación (1%)
MIN_CONFIDENCE = 0.50     # Confianza mínima
MIN_RR = 1.5              # Ratio R:R mínimo
LIVE_MODE = True          # True = loop continuo
UPDATE_INTERVAL = 15      # Minutos entre escaneos
```

## 📈 Uso

```bash
python main.py
```

### Modo único (una ejecución)
```python
LIVE_MODE = False
```

### Modo Live (24/5)
```python
LIVE_MODE = True
UPDATE_INTERVAL = 15
```
Presiona `Ctrl+C` para detener.

## 📊 Output ejemplo

```
🏆 TOP 4 OPORTUNIDADES (ordenadas por confianza)
══════════════════════════════════════════════════════════════════════════════════════════
  #   Par        Acción   Conf   Grade   Bias           Entry      SL         TP         Lots
  -------------------------------------------------------------------------------------------
  1   USDJPY     🟢BUY    1.00   A(86)   Alcista claro  157.265    156.977    157.778    0.06
  2   EURCHF     🟢BUY    1.00   B(68)   Alcista claro  0.93187    0.93077    0.93451    0.10
  3   USDCHF     🟢BUY    0.86   B(68)   Alcista claro  0.79961    0.79848    0.80328    0.08
```

## ⚠️ Disclaimer

Este software es solo para fines educativos. No es asesoramiento financiero. Opera bajo tu propio riesgo.

## 📝 Licencia

MIT
