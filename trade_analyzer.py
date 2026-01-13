import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# --- CONFIG ---
st.set_page_config(page_title="God Mode: Sniper Advisor", page_icon="🎯", layout="centered")

# --- CLASS DEFINITION (Sama seperti sebelumnya, dimodifikasi outputnya) ---
class GodModeWeb:
    def __init__(self, ticker):
        self.clean_ticker = ticker.upper().replace('.JK', '')
        self.ticker = self.clean_ticker + '.JK'
        self.df = None

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period="5d", interval="5m", progress=False)
            if self.df.empty: return "EMPTY"
            if len(self.df) < 50: return "FEW_DATA"
            self.df.reset_index(inplace=True)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)
            return "OK"
        except Exception as e:
            return str(e)

    def round_price(self, price):
        if price < 200: tick = 1
        elif price < 500: tick = 2
        elif price < 2000: tick = 5
        elif price < 5000: tick = 10
        else: tick = 25
        return round(price / tick) * tick

    def analyze(self):
        # (LOGIC HITUNGAN SAMA PERSIS DENGAN SEBELUMNYA)
        df = self.df
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['TPV'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']
        df['VWAP'] = df['TPV'].cumsum() / df['Volume'].cumsum()
        
        # ADX Manual Calc
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['up_move'] = df['High'] - df['High'].shift(1)
        df['down_move'] = df['Low'].shift(1) - df['Low']
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        df['TR_smooth'] = df['TR'].ewm(alpha=1/14, adjust=False).mean()
        df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=1/14, adjust=False).mean()
        df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=1/14, adjust=False).mean()
        df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['TR_smooth'])
        df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['TR_smooth'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['ADX'] = df['dx'].ewm(alpha=1/14, adjust=False).mean()

        # Indicators
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        min_rsi = df['RSI'].rolling(14).min()
        max_rsi = df['RSI'].rolling(14).max()
        df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
        
        df['HighLow'] = df['High'] - df['Low']
        df['ATR'] = df['HighLow'].rolling(14).mean()
        
        df.dropna(inplace=True)
        row = df.iloc[-1]
        
        # S/R 75 Candle
        recent_high = df['High'].tail(75).max()
        recent_low = df['Low'].tail(75).min()
        res_price = self.round_price(recent_high)
        sup_price = self.round_price(recent_low)
        volatility_pct = ((recent_high - recent_low) / recent_low) * 100

        p = row['Close']
        score = 50
        reasons = []

        # Zombie Filter
        is_zombie = False
        if volatility_pct < 2.0:
            is_zombie = True
            score = 0
            reasons.append("⛔ SAHAM TIDUR (Gerak < 2% Seharian)")
        else:
            reasons.append(f"✅ Volatilitas Sehat ({volatility_pct:.2f}%)")

        if not is_zombie:
            if p > row['MA20']: score += 10; reasons.append("✅ Bullish Trend (> MA20)")
            else: score -= 15; reasons.append("❌ Bearish Trend (< MA20)")
            
            if p > row['VWAP']: score += 10; reasons.append("✅ Buyer Dominan (> VWAP)")
            else: score -= 10; reasons.append("❌ Seller Dominan (< VWAP)")
            
            if row['ADX'] > 25: 
                if p > row['MA20']: score += 10; reasons.append("✅ Power Kuat (ADX > 25)")
                else: score -= 10; reasons.append("❌ Tekanan Jual Kuat")
            
            if row['StochRSI'] < 0.2: score += 10; reasons.append("✅ Oversold (Murah)")
            elif row['StochRSI'] > 0.8: score -= 5; reasons.append("⚠️ Overbought (Mahal)")
            
            if p >= res_price: score += 5; reasons.append("🚀 BREAKOUT RESISTANCE!")

        score = max(0, min(score, 100))
        
        # Trade Plan
        sl_dist = row['ATR'] * 1.5
        raw_sl = p - sl_dist
        if (p - raw_sl) / p < 0.01: raw_sl = p * 0.985
        sl_final = self.round_price(raw_sl)
        if sl_final >= p: sl_final = p - (2 if p<200 else 5)
        
        risk = p - sl_final
        tp1 = self.round_price(p + (risk * 1.5))
        tp2 = self.round_price(p + (risk * 3.0))
        tp3 = self.round_price(p + (risk * 5.0))

        # Strategi
        if is_zombie: strategy = "AVOID"
        elif score >= 75: strategy = "GAS POL (Aggressive)"
        elif score >= 60: strategy = "BUY ON WEAKNESS"
        elif score >= 40: strategy = "WAIT / SPECULATIVE"
        else: strategy = "AVOID / SELL"

        return {
            'data': row, 'score': score, 'reasons': reasons,
            'sl': sl_final, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'strategy': strategy, 'support': sup_price, 'resistance': res_price,
            'is_zombie': is_zombie, 'volatility': volatility_pct
        }

# --- UI STREAMLIT ---
st.title("🎯 God Mode: Sniper Advisor")
st.caption("Brutal Honest Trading Assistant")

ticker_input = st.text_input("Masukkan Kode Saham (Contoh: BBRI, ANTM)", "").upper()

if st.button("Analisa Sekarang"):
    if not ticker_input:
        st.error("Masukkan kode saham dulu!")
    else:
        with st.spinner(f"Mengaudit {ticker_input}..."):
            bot = GodModeWeb(ticker_input)
            status = bot.fetch_data()
            
            if status == "OK":
                res = bot.analyze()
                d = res['data']
                p = d['Close']

                # 1. Header Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Harga Last", f"{p:.0f}")
                col2.metric("Skor Kualitas", f"{res['score']}/100")
                col3.metric("Status", res['strategy'])

                # 2. Peta Perang
                st.subheader("🛡️ Peta Perang (Intraday)")
                c1, c2, c3 = st.columns(3)
                
                # Logic warna Resistance
                if p >= res['resistance']:
                    c1.warning(f"Resist: {res['resistance']:.0f} (BREAKOUT!)")
                else:
                    c1.info(f"Resist: {res['resistance']:.0f}")
                    
                c2.success(f"Support: {res['support']:.0f}")
                c3.info(f"Volatilitas: {res['volatility']:.2f}%")

                # 3. Indikator
                st.markdown("---")
                st.write("**Data Indikator:**")
                st.text(f"VWAP: {d['VWAP']:.0f} | MA5: {d['MA5']:.0f} | ADX: {d['ADX']:.1f}")

                # 4. Alasan
                st.markdown("---")
                st.write("**Audit Log:**")
                for r in res['reasons']:
                    st.write(f"- {r}")

                # 5. Trade Plan
                if not res['is_zombie'] and "AVOID" not in res['strategy']:
                    st.markdown("---")
                    st.subheader("📋 Trade Plan")
                    
                    st.success(f"🛒 **ENTRY AREA:** {bot.round_price(min(p, d['MA5'])):.0f} - {bot.round_price(max(p, d['MA5'])):.0f}")
                    st.error(f"🛡️ **STOP LOSS:** {res['sl']:.0f}")
                    
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Target 1", f"{res['tp1']:.0f}")
                    t2.metric("Target 2", f"{res['tp2']:.0f}")
                    t3.metric("Target 3", f"{res['tp3']:.0f}")
                    
                    st.warning("⚠️ **VALIDASI MANUSIA:** Cek Order Book & Bandarmology sebelum Entry!")
                elif res['is_zombie']:
                    st.error("⛔ SAHAM ZOMBIE. Cari yang lain.")
                else:
                    st.error("⛔ JANGAN MASUK. Tren Jelek.")

            elif status == "EMPTY":
                st.error("Data saham tidak ditemukan / Error Koneksi.")
            elif status == "FEW_DATA":
                st.error("Data saham terlalu sedikit (Baru IPO?).")
            else:
                st.error(f"Error System: {status}")