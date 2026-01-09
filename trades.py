# -*- coding: utf-8 -*-
"""
Trade Tracking System
Gestión de operaciones abiertas y cerradas con seguimiento en tiempo real.

Comandos:
    python trades.py --open 1       # Abrir trade #1 del último escaneo
    python trades.py --close 1      # Cerrar trade #1 (te pregunta resultado)
    python trades.py --list         # Ver trades abiertos con seguimiento
    python trades.py --history      # Ver historial de trades cerrados
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional
import yfinance as yf

# Archivos de datos
TRADES_FILE = os.path.join(os.path.dirname(__file__), "trades.json")
SIGNALS_FILE = os.path.join(os.path.dirname(__file__), "last_signals.json")


# ═══════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════

def load_json(filepath: str) -> dict:
    """Carga un archivo JSON, retorna dict vacío si no existe."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(filepath: str, data: dict):
    """Guarda datos en archivo JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def get_current_price(ticker: str) -> Optional[float]:
    """Obtiene el precio actual de un par."""
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if data is not None and len(data) > 0:
            if hasattr(data.columns, 'get_level_values'):
                data.columns = data.columns.get_level_values(0)
            return float(data["Close"].iloc[-1])
    except Exception:
        pass
    return None


def pip_size(ticker: str) -> float:
    """Retorna el tamaño de pip para un par."""
    return 0.01 if "JPY" in ticker.upper() else 0.0001


def calculate_pips(ticker: str, entry: float, current: float, action: str) -> float:
    """Calcula pips de ganancia/pérdida."""
    pip = pip_size(ticker)
    diff = current - entry
    if action == "SELL":
        diff = -diff
    return round(diff / pip, 1)


def get_session_info() -> Dict[str, object]:
    """Obtiene info de la sesión actual."""
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    
    sessions = {
        "Sydney": {"start": 21, "end": 6},
        "Tokyo": {"start": 23, "end": 8},
        "London": {"start": 8, "end": 17},
        "New York": {"start": 13, "end": 22},
    }
    
    active = []
    for name, times in sessions.items():
        start, end = times["start"], times["end"]
        if start > end:
            is_active = hour >= start or hour < end
        else:
            is_active = start <= hour < end
        if is_active:
            active.append(name)
    
    return {
        "hour_utc": hour,
        "active_sessions": active,
        "sessions_info": sessions
    }


def time_until_session_end(opened_session: str) -> str:
    """Calcula tiempo hasta que cierre una sesión."""
    session_ends = {
        "Sydney": 6,
        "Tokyo": 8,
        "London": 17,
        "New York": 22,
    }
    
    if opened_session not in session_ends:
        return "N/A"
    
    now_utc = datetime.now(timezone.utc)
    end_hour = session_ends[opened_session]
    
    hours_left = end_hour - now_utc.hour
    if hours_left < 0:
        hours_left += 24
    
    mins_left = 60 - now_utc.minute
    if mins_left == 60:
        mins_left = 0
    else:
        hours_left -= 1
    
    if hours_left < 0:
        hours_left = 0
        
    return f"{hours_left}h {mins_left}m"


# ═══════════════════════════════════════════════════════════════
# GESTIÓN DE TRADES
# ═══════════════════════════════════════════════════════════════

def load_trades() -> Dict[str, List[dict]]:
    """Carga los trades desde el archivo JSON."""
    data = load_json(TRADES_FILE)
    if "open" not in data:
        data["open"] = []
    if "closed" not in data:
        data["closed"] = []
    return data


def save_trades(data: Dict[str, List[dict]]):
    """Guarda los trades en el archivo JSON."""
    save_json(TRADES_FILE, data)


def load_last_signals() -> List[dict]:
    """Carga las señales del último escaneo."""
    data = load_json(SIGNALS_FILE)
    return data.get("signals", [])


def save_last_signals(signals: List[dict]):
    """Guarda las señales del último escaneo."""
    save_json(SIGNALS_FILE, {
        "timestamp": datetime.now().isoformat(),
        "signals": signals
    })


def open_trade(signal_num: int) -> bool:
    """
    Abre un trade basado en el número de señal del último escaneo.
    """
    signals = load_last_signals()
    
    if not signals:
        print("\n❌ No hay señales guardadas del último escaneo.")
        print("   Ejecuta primero: python main.py")
        return False
    
    if signal_num < 1 or signal_num > len(signals):
        print(f"\n❌ Número inválido. Señales disponibles: 1-{len(signals)}")
        return False
    
    signal = signals[signal_num - 1]
    
    # Extraer datos de la señal
    ticker = signal.get("ticker", "")
    action = signal.get("summary", {}).get("action", "")
    risk = signal.get("risk", {})
    position = signal.get("position", {})
    quality = signal.get("quality", {})
    spread_adj = signal.get("spread_adjustment", {})
    
    entry = risk.get("entry", 0)
    sl = spread_adj.get("sl_adjusted", risk.get("sl", 0))
    tp = spread_adj.get("tp_adjusted", risk.get("tp", 0))
    lots = position.get("lots", 0)
    
    # Obtener sesión actual
    session_info = get_session_info()
    opened_in_session = session_info["active_sessions"][0] if session_info["active_sessions"] else "Unknown"
    
    # Crear trade
    trade = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "ticker": ticker,
        "symbol": ticker.replace("=X", ""),
        "action": action,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "sl_original": risk.get("sl", 0),
        "tp_original": risk.get("tp", 0),
        "lots": lots,
        "grade": quality.get("grade", "?"),
        "score": quality.get("score", 0),
        "opened_at": datetime.now().isoformat(),
        "opened_in_session": opened_in_session,
        "breakeven_moved": False,
        "partial_closed": False,
        "notes": ""
    }
    
    # Guardar
    trades_data = load_trades()
    trades_data["open"].append(trade)
    save_trades(trades_data)
    
    decimals = 3 if "JPY" in ticker else 5
    
    print(f"\n{'═' * 60}")
    print(f"  ✅ TRADE ABIERTO")
    print(f"{'═' * 60}")
    print(f"  Par: {trade['symbol']}")
    print(f"  Acción: {'🟢 BUY' if action == 'BUY' else '🔴 SELL'}")
    print(f"  Entry: {entry:.{decimals}f}")
    print(f"  SL: {sl:.{decimals}f} (ajustado por spread)")
    print(f"  TP: {tp:.{decimals}f} (ajustado por spread)")
    print(f"  Lotes: {lots:.2f}")
    print(f"  Grado: {trade['grade']} ({trade['score']})")
    print(f"  Sesión: {opened_in_session}")
    print(f"{'═' * 60}\n")
    
    return True


def close_trade(trade_num: int, result: str = None) -> bool:
    """
    Cierra un trade abierto.
    result: 'win', 'loss', 'be' (breakeven), o None para preguntar
    """
    trades_data = load_trades()
    open_trades = trades_data.get("open", [])
    
    if not open_trades:
        print("\n❌ No hay trades abiertos.")
        return False
    
    if trade_num < 1 or trade_num > len(open_trades):
        print(f"\n❌ Número inválido. Trades abiertos: 1-{len(open_trades)}")
        return False
    
    trade = open_trades[trade_num - 1]
    
    # Obtener precio actual
    current_price = get_current_price(trade["ticker"])
    pips = 0
    if current_price:
        pips = calculate_pips(trade["ticker"], trade["entry"], current_price, trade["action"])
    
    # Si no se especificó resultado, preguntar
    if result is None:
        print(f"\n  Trade: {trade['symbol']} {trade['action']}")
        print(f"  Entry: {trade['entry']} | Actual: {current_price} | P&L: {pips:+.1f} pips")
        print(f"\n  ¿Cómo cerró?")
        print(f"    1. TP alcanzado (win)")
        print(f"    2. SL tocado (loss)")
        print(f"    3. Breakeven (be)")
        print(f"    4. Cierre manual")
        print(f"    0. Cancelar")
        
        try:
            choice = input("\n  > ").strip()
        except EOFError:
            return False
            
        result_map = {"1": "win", "2": "loss", "3": "be", "4": "manual"}
        if choice == "0":
            print("  Cancelado.")
            return False
        result = result_map.get(choice, "manual")
    
    # Calcular P&L final
    pip = pip_size(trade["ticker"])
    if result == "win":
        pips_final = abs(trade["tp"] - trade["entry"]) / pip
    elif result == "loss":
        pips_final = -abs(trade["sl"] - trade["entry"]) / pip
    elif result == "be":
        pips_final = 0
    else:
        pips_final = pips  # Usar P&L actual
    
    # Mover a cerrados
    trade["closed_at"] = datetime.now().isoformat()
    trade["result"] = result
    trade["pips_final"] = round(pips_final, 1)
    trade["exit_price"] = current_price
    
    open_trades.pop(trade_num - 1)
    trades_data["closed"].append(trade)
    save_trades(trades_data)
    
    emoji = "✅" if result == "win" else ("❌" if result == "loss" else "⏹️")
    print(f"\n{emoji} Trade cerrado: {trade['symbol']} {trade['action']} → {result.upper()} ({pips_final:+.1f} pips)")
    
    return True


def list_open_trades(show_alerts: bool = True):
    """
    Muestra todos los trades abiertos con seguimiento en tiempo real.
    """
    trades_data = load_trades()
    open_trades = trades_data.get("open", [])
    
    if not open_trades:
        print("\n📊 No hay trades abiertos.\n")
        return
    
    print(f"\n{'═' * 80}")
    print(f"  📊 TUS OPERACIONES ABIERTAS ({len(open_trades)})")
    print(f"{'═' * 80}")
    print(f"  {'#':<3} {'Par':<8} {'Dir':<5} {'Entry':<10} {'Actual':<10} {'P&L':<12} {'SL':<10} {'TP':<10} {'Estado':<15}")
    print(f"  {'-' * 77}")
    
    alerts = []
    total_pips = 0
    winning = 0
    losing = 0
    
    for i, trade in enumerate(open_trades, 1):
        ticker = trade["ticker"]
        symbol = trade["symbol"]
        action = trade["action"]
        entry = trade["entry"]
        sl = trade["sl"]
        tp = trade["tp"]
        
        # Obtener precio actual
        current = get_current_price(ticker)
        if current is None:
            current = entry
            pips = 0
            status = "⚠️ Sin datos"
        else:
            pips = calculate_pips(ticker, entry, current, action)
            total_pips += pips
            
            if pips > 0:
                winning += 1
            elif pips < 0:
                losing += 1
            
            # Calcular estado
            pip = pip_size(ticker)
            sl_dist = abs(entry - sl) / pip
            r1_target = sl_dist  # 1R en pips
            
            if pips >= r1_target and not trade.get("breakeven_moved"):
                status = "✅ +1R alcanzado"
                alerts.append(f"🔔 {symbol}: Alcanzó +1R → MOVER SL a breakeven ({entry})")
            elif pips >= r1_target * 0.5:
                status = "📈 +0.5R"
            elif pips <= -sl_dist * 0.7:
                status = "⚠️ Cerca de SL"
            elif pips < 0:
                status = "📉 En pérdida"
            else:
                status = "⏳ En rango"
        
        decimals = 3 if "JPY" in symbol else 5
        pips_str = f"{pips:+.1f} pips"
        
        print(f"  {i:<3} {symbol:<8} {action:<5} {entry:<10.{decimals}f} {current:<10.{decimals}f} {pips_str:<12} {sl:<10.{decimals}f} {tp:<10.{decimals}f} {status:<15}")
        
        # Alerta de sesión
        session_info = get_session_info()
        opened_session = trade.get("opened_in_session", "")
        if opened_session and opened_session not in session_info["active_sessions"]:
            alerts.append(f"⏰ {symbol}: Sesión {opened_session} ya cerró → Considerar cerrar")
        elif opened_session:
            time_left = time_until_session_end(opened_session)
            if "0h" in time_left:
                alerts.append(f"⏰ {symbol}: Sesión {opened_session} cierra en {time_left}")
    
    print(f"  {'-' * 77}")
    
    # Mostrar alertas
    if show_alerts and alerts:
        print(f"\n  📋 ALERTAS:")
        for alert in alerts:
            print(f"  {alert}")
    
    # Resumen
    print(f"\n  📈 RESUMEN:")
    print(f"  Total P&L abierto: {total_pips:+.1f} pips")
    print(f"  Ganando: {winning} | Perdiendo: {losing}")
    print(f"{'═' * 80}\n")


def show_history(limit: int = 20):
    """Muestra historial de trades cerrados."""
    trades_data = load_trades()
    closed_trades = trades_data.get("closed", [])
    
    if not closed_trades:
        print("\n📜 No hay historial de trades.\n")
        return
    
    # Ordenar por fecha (más recientes primero)
    closed_trades = sorted(closed_trades, key=lambda x: x.get("closed_at", ""), reverse=True)
    
    print(f"\n{'═' * 70}")
    print(f"  📜 HISTORIAL DE TRADES (últimos {min(limit, len(closed_trades))})")
    print(f"{'═' * 70}")
    print(f"  {'Fecha':<12} {'Par':<8} {'Dir':<5} {'Resultado':<10} {'Pips':<10} {'Grado':<6}")
    print(f"  {'-' * 55}")
    
    total_pips = 0
    wins = 0
    losses = 0
    
    for trade in closed_trades[:limit]:
        closed_at = trade.get("closed_at", "")[:10]
        symbol = trade.get("symbol", "")
        action = trade.get("action", "")
        result = trade.get("result", "")
        pips = trade.get("pips_final", 0)
        grade = trade.get("grade", "?")
        
        total_pips += pips
        if result == "win":
            wins += 1
            result_str = "✅ WIN"
        elif result == "loss":
            losses += 1
            result_str = "❌ LOSS"
        else:
            result_str = "⏹️ " + result.upper()
        
        print(f"  {closed_at:<12} {symbol:<8} {action:<5} {result_str:<10} {pips:+.1f}{'':>4} {grade:<6}")
    
    print(f"  {'-' * 55}")
    print(f"\n  📊 ESTADÍSTICAS:")
    total_trades = wins + losses
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"  Winrate: {winrate:.1f}% ({wins}W / {losses}L)")
    print(f"  Total pips: {total_pips:+.1f}")
    print(f"{'═' * 70}\n")


def mark_breakeven(trade_num: int) -> bool:
    """Marca que se movió el SL a breakeven."""
    trades_data = load_trades()
    open_trades = trades_data.get("open", [])
    
    if trade_num < 1 or trade_num > len(open_trades):
        print(f"❌ Trade #{trade_num} no encontrado.")
        return False
    
    open_trades[trade_num - 1]["breakeven_moved"] = True
    open_trades[trade_num - 1]["sl"] = open_trades[trade_num - 1]["entry"]
    save_trades(trades_data)
    
    symbol = open_trades[trade_num - 1]["symbol"]
    print(f"✅ {symbol}: SL movido a breakeven")
    return True


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de tracking de trades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python trades.py --list           Ver trades abiertos
  python trades.py --open 1         Abrir trade #1 del último escaneo
  python trades.py --close 1        Cerrar trade #1
  python trades.py --be 1           Marcar breakeven en trade #1
  python trades.py --history        Ver historial
        """
    )
    
    parser.add_argument("--open", "-o", type=int, metavar="N",
                        help="Abrir trade #N del último escaneo")
    parser.add_argument("--close", "-c", type=int, metavar="N",
                        help="Cerrar trade #N")
    parser.add_argument("--list", "-l", action="store_true",
                        help="Ver trades abiertos con seguimiento")
    parser.add_argument("--history", "-H", action="store_true",
                        help="Ver historial de trades cerrados")
    parser.add_argument("--be", type=int, metavar="N",
                        help="Marcar breakeven en trade #N")
    parser.add_argument("--signals", "-s", action="store_true",
                        help="Ver señales del último escaneo")
    
    args = parser.parse_args()
    
    # Si no hay argumentos, mostrar lista
    if len(sys.argv) == 1:
        list_open_trades()
        return
    
    if args.open:
        open_trade(args.open)
    elif args.close:
        close_trade(args.close)
    elif args.list:
        list_open_trades()
    elif args.history:
        show_history()
    elif args.be:
        mark_breakeven(args.be)
    elif args.signals:
        signals = load_last_signals()
        if not signals:
            print("\n❌ No hay señales guardadas.")
        else:
            print(f"\n📊 SEÑALES DEL ÚLTIMO ESCANEO ({len(signals)}):")
            for i, sig in enumerate(signals, 1):
                symbol = sig.get("ticker", "").replace("=X", "")
                action = sig.get("summary", {}).get("action", "")
                grade = sig.get("quality", {}).get("grade", "?")
                entry = sig.get("risk", {}).get("entry", 0)
                decimals = 3 if "JPY" in symbol else 5
                print(f"  {i}. {symbol} {action} @ {entry:.{decimals}f} (Grado {grade})")
            print()


if __name__ == "__main__":
    main()
