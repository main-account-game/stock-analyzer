import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 🎨 KONFIGURASI HALAMAN (WAJIB DI ATAS)
# ==========================================
st.set_page_config(
    page_title="GOD MODE: SNIPER",
    page_icon="🎯",
    layout="wide", # Pakai seluruh lebar layar
    initial_sidebar_state="expanded"
)

# Custom CSS untuk Dark Mode & Card Style
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .big-font { font-size: 24px !important; font-weight: bold; }
    .med-font { font-size: 18px !important; }
    .green-text { color: #00ff00; }
    .red-text { color: #ff4b4b; }
    .yellow-text { color: #ffeb3b; }
    .magenta-text { color: #d500f9; }
    
    /* Highlight Box untuk Rekomendasi */
    .rec-box-buy {
        background-color: #06402b;
        color: #00ff00;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #00ff00;
    }
    .rec-box-sell {
        background-color: #4a0d0d;
        color: #ff4b4b;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #ff4b4b;
    }
    .rec-box-wait {
        background-color: #423d08;
        color: #ffeb3b;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #ffeb3b;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 LOGIC ENGINE (SAMA SEPERTI CLI)
# ==========================================
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
        df = self.df
        # Indikator
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['TPV'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']
        df['VWAP'] = df['TPV'].cumsum() / df['Volume'].cumsum()
        
        # ADX Manual
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

        # StochRSI & ATR
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
        
        # 75 Candle S/R
        recent_high = df['High'].tail(75).max()
        recent_low = df['Low'].tail(75).min()
        res_price = self.round_price(recent_high)
        sup_price = self.round_price(recent_low)
        volatility_pct = ((recent_high - recent_low) / recent_low) * 100

        p = row['Close']
        
        # SCORING
        score = 50
        reasons = []

        is_zombie = False
        if volatility_pct < 2.0:
            is_zombie = True
            score = 0
            reasons.append("⛔ SAHAM ZOMBIE (Gerak < 2%)")
        
        if not is_zombie:
            if p > row['MA20']: score += 10; reasons.append("✅ Trend Bullish (> MA20)")
            else: score -= 15; reasons.append("❌ Trend Bearish (< MA20)")
            
            if p > row['VWAP']: score += 10; reasons.append("✅ Dominasi Buyer (> VWAP)")
            else: score -= 10; reasons.append("❌ Dominasi Seller (< VWAP)")
            
            if row['ADX'] > 25: 
                if p > row['MA20']: score += 10; reasons.append("✅ Power Kuat (ADX > 25)")
                else: score -= 10; reasons.append("❌ Jualan Kuat")
            
            if row['StochRSI'] < 0.2: score += 10; reasons.append("✅ Oversold (Murah)")
            elif row['StochRSI'] > 0.8: score -= 5; reasons.append("⚠️ Overbought (Mahal)")
            
            if p >= res_price: score += 5; reasons.append("🚀 BREAKOUT DAY HIGH!")

        score = max(0, min(score, 100))
        
        # PLAN
        sl_dist = row['ATR'] * 1.5
        raw_sl = p - sl_dist
        if (p - raw_sl) / p < 0.01: raw_sl = p * 0.985
        sl_final = self.round_price(raw_sl)
        if sl_final >= p: sl_final = p - (2 if p<200 else 5)
        
        risk = p - sl_final
        risk_pct = (risk/p)*100
        tp1 = self.round_price(p + (risk * 1.5))
        tp2 = self.round_price(p + (risk * 3.0))
        tp3 = self.round_price(p + (risk * 5.0))

        if is_zombie: 
            rec_text = "AVOID / ZOMBIE"
            rec_style = "rec-box-sell"
        elif score >= 75: 
            rec_text = "🚀 GAS POL / AGGRESSIVE"
            rec_style = "rec-box-buy"
        elif score >= 60: 
            rec_text = "✅ BUY ON WEAKNESS"
            rec_style = "rec-box-buy"
        elif score >= 40: 
            rec_text = "⚠️ SPECULATIVE / WAIT"
            rec_style = "rec-box-wait"
        else: 
            rec_text = "⛔ AVOID / SELL"
            rec_style = "rec-box-sell"

        return {
            'data': row, 'score': score, 'reasons': reasons,
            'sl': sl_final, 'risk_pct': risk_pct,
            'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'rec_text': rec_text, 'rec_style': rec_style,
            'support': sup_price, 'resistance': res_price,
            'is_zombie': is_zombie, 'volatility': volatility_pct
        }

# ==========================================
# 🖥️ TAMPILAN DASHBOARD (UI CODE)
# ==========================================

# Sidebar
with st.sidebar:
    st.title("🎛️ CONTROL PANEL")
    ticker_input = st.text_input("Kode Saham", "").upper()
    analyze_btn = st.button("🚀 ANALISA SEKARANG", type="primary")
    st.markdown("---")
    st.info("💡 **Tips:** Gunakan jam 09:15 - 11:30 untuk hasil terbaik.")

if analyze_btn and ticker_input:
    bot = GodModeWeb(ticker_input)
    
    with st.spinner('📡 Mengambil data...'):
        status = bot.fetch_data()

    if status == "OK":
        res = bot.analyze()
        d = res['data']
        p = d['Close']
        
        # --- HEADER SECTION ---
        col_h1, col_h2 = st.columns([1, 3])
        with col_h1:
            st.metric("HARGA SAAT INI", f"{p:.0f}")
        with col_h2:
            st.markdown(f"""
            <div class="{res['rec_style']}">
                <div class="big-font">{res['rec_text']}</div>
                <div class="med-font">SKOR KUALITAS: {res['score']}/100</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")

        # --- TECHNICAL COCKPIT (GRID) ---
        st.subheader("📊 DASHBOARD INDIKATOR")
        
        # Logic Warna Text
        trend_clr = "green-text" if p > d['MA20'] else "red-text"
        vwap_clr = "green-text" if p > d['VWAP'] else "red-text"
        adx_clr = "green-text" if d['ADX'] > 25 else "yellow-text"
        stoch_clr = "green-text" if d['StochRSI'] < 0.2 else ("red-text" if d['StochRSI'] > 0.8 else "yellow-text")

        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <small>TREND (MA20)</small><br>
                <span class="big-font {trend_clr}">{d['MA20']:.0f}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <small>BANDAR (VWAP)</small><br>
                <span class="big-font {vwap_clr}">{d['VWAP']:.0f}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <small>POWER (ADX)</small><br>
                <span class="big-font {adx_clr}">{d['ADX']:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <small>TIMING (STOCH)</small><br>
                <span class="big-font {stoch_clr}">{d['StochRSI']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

        # --- BATTLEFIELD MAP ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⚔️ S/R INTRADAY")
        
        b1, b2, b3 = st.columns(3)
        with b1:
             st.markdown(f"""
            <div class="metric-card" style="border-color: #ff4b4b;">
                <small style="color:#ff4b4b">RESISTANCE (ATAP)</small><br>
                <span class="big-font">{res['resistance']:.0f}</span>
            </div>
            """, unsafe_allow_html=True)
        with b2:
             st.markdown(f"""
            <div class="metric-card" style="border-color: #00ff00;">
                <small style="color:#00ff00">SUPPORT (LANTAI)</small><br>
                <span class="big-font">{res['support']:.0f}</span>
            </div>
            """, unsafe_allow_html=True)
        with b3:
             st.markdown(f"""
            <div class="metric-card">
                <small>VOLATILITAS</small><br>
                <span class="big-font">{res['volatility']:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # --- TRADE PLAN SECTION ---
        if not res['is_zombie'] and "AVOID" not in res['rec_text']:
            st.markdown("---")
            st.subheader("📋 TRADE PLAN")
            
            entry_min = bot.round_price(min(p, d['MA5']))
            entry_max = bot.round_price(max(p, d['MA5']))
            
            # Tampilan Plan dalam Kolom
            p1, p2 = st.columns([1, 1])
            
            with p1:
                st.markdown(f"""
                <div style="background:#0e1117; padding:15px; border-radius:10px; border-left: 5px solid cyan;">
                    <h4 style="margin:0; color:cyan;">🛒 ENTRY AREA</h4>
                    <span class="big-font">{entry_min:.0f} - {entry_max:.0f}</span>
                </div>
                <br>
                <div style="background:#0e1117; padding:15px; border-radius:10px; border-left: 5px solid red;">
                    <h4 style="margin:0; color:#ff4b4b;">🛡️ STOP LOSS</h4>
                    <span class="big-font">{res['sl']:.0f}</span> <small>(Risk: -{res['risk_pct']:.2f}%)</small>
                </div>
                """, unsafe_allow_html=True)
                
            with p2:
                # Menghitung Gain Persen
                g1 = ((res['tp1'] - p) / p) * 100
                g2 = ((res['tp2'] - p) / p) * 100
                g3 = ((res['tp3'] - p) / p) * 100
                
                st.markdown(f"""
                <div style="background:#0e1117; padding:10px; border-radius:5px; margin-bottom:5px;">
                    <span style="color:#00ff00;">🎯 TARGET 1:</span> <b>{res['tp1']:.0f}</b> (+{g1:.1f}%)
                </div>
                <div style="background:#0e1117; padding:10px; border-radius:5px; margin-bottom:5px;">
                    <span style="color:#00ff00;">🚀 TARGET 2:</span> <b>{res['tp2']:.0f}</b> (+{g2:.1f}%)
                </div>
                <div style="background:#0e1117; padding:10px; border-radius:5px; margin-bottom:5px;">
                    <span style="color:#d500f9;">💎 JACKPOT :</span> <b>{res['tp3']:.0f}</b> (+{g3:.1f}%)
                </div>
                """, unsafe_allow_html=True)
                
            # VALIDASI AKHIR
            st.warning("⚠️ **VALIDASI MANUSIA:** Cek Order Book (Bid Tebal) & Broker Summary (Akumulasi) sebelum Entry!")

        elif res['is_zombie']:
            st.error("⛔ SAHAM ZOMBIE DETECTED: Jangan entry, fee transaksi akan memakan modal Anda.")
            
        else:
            st.error("⛔ SETUP TIDAK VALID: Tunggu harga naik ke atas MA20 atau VWAP.")

        # --- AUDIT LOG ---
        with st.expander("🔍 Lihat Detail Audit Skor (Kenapa angka ini muncul?)"):
            for r in res['reasons']:
                st.write(r)

    elif status == "EMPTY":
        st.error("❌ Data Tidak Ditemukan. Cek kode saham.")
    elif status == "FEW_DATA":
        st.error("❌ Data Terlalu Sedikit (Saham Baru IPO).")
    else:
        st.error(f"❌ Error: {status}")

else:
    st.info("👈 Masukkan Kode Saham di Sidebar sebelah kiri untuk memulai.")
