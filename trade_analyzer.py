import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ==========================================
# 1. KONFIGURASI HALAMAN (DARK MODE FORCE)
# ==========================================
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    # layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Paksa Background Gelap */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Card Style */
    .metric-card {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Typography */
    .big-font { font-size: 20px !important; font-weight: bold; }
    .huge-font { font-size: 32px !important; font-weight: bold; }
    .label-font { font-size: 12px !important; color: #A0A0A0; }
    
    /* Colors matches Colorama Logic */
    .c-green { color: #00FF00 !important; }
    .c-red { color: #FF4B4B !important; }
    .c-yellow { color: #FFEB3B !important; }
    .c-magenta { color: #D500F9 !important; }
    .c-cyan { color: #00E5FF !important; }
    
    /* Status Boxes */
    .box-buy { border: 2px solid #00FF00; background-color: #002200; padding: 20px; border-radius: 10px; text-align:center; }
    .box-sell { border: 2px solid #FF4B4B; background-color: #220000; padding: 20px; border-radius: 10px; text-align:center; }
    .box-wait { border: 2px solid #FFEB3B; background-color: #222200; padding: 20px; border-radius: 10px; text-align:center; }
    
    /* Trade Plan Box */
    .plan-box {
        background-color: #161920;
        border-left: 4px solid #00E5FF;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    /* Audit Box */
    .audit-box {
        background-color: #13161c;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC ENGINE
# ==========================================
class GodModeEngine:
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

    def calculate_adx(self, window=14):
        df = self.df.copy()
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        df['up_move'] = df['High'] - df['High'].shift(1)
        df['down_move'] = df['Low'].shift(1) - df['Low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        df['TR_smooth'] = df['TR'].ewm(alpha=1/window, adjust=False).mean()
        df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=1/window, adjust=False).mean()
        df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=1/window, adjust=False).mean()
        
        df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['TR_smooth'])
        df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['TR_smooth'])
        
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['ADX'] = df['dx'].ewm(alpha=1/window, adjust=False).mean()
        return df['ADX']

    def analyze_market(self):
        self.df['MA5'] = self.df['Close'].rolling(5).mean()
        self.df['MA20'] = self.df['Close'].rolling(20).mean()
        self.df['TPV'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3 * self.df['Volume']
        self.df['VWAP'] = self.df['TPV'].cumsum() / self.df['Volume'].cumsum()
        
        exp12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp12 - exp26
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['Hist'] = self.df['MACD'] - self.df['Signal']

        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        min_rsi = self.df['RSI'].rolling(14).min()
        max_rsi = self.df['RSI'].rolling(14).max()
        self.df['StochRSI'] = (self.df['RSI'] - min_rsi) / (max_rsi - min_rsi)

        self.df['ADX'] = self.calculate_adx()
        self.df['HighLow'] = self.df['High'] - self.df['Low']
        self.df['ATR'] = self.df['HighLow'].rolling(14).mean()

        self.df.dropna(inplace=True)
        row = self.df.iloc[-1]
        
        recent_high = self.df['High'].tail(75).max()
        recent_low = self.df['Low'].tail(75).min()
        res_price = self.round_price(recent_high)
        sup_price = self.round_price(recent_low)
        
        volatility_pct = ((recent_high - recent_low) / recent_low) * 100
        
        p = row['Close']
        score = 50
        reasons = []

        is_zombie = False
        if volatility_pct < 2.0:
            is_zombie = True
            score = 0 
            reasons.append("⛔ [BAHAYA] SAHAM TIDUR! Range gerak < 2%")
        else:
            reasons.append(f"✅ [INFO] Volatilitas Sehat ({volatility_pct:.1f}%)")

        if not is_zombie:
            if p > row['MA20']: score += 10; reasons.append("✅ [TREND] Bullish (Harga > MA20)")
            else: score -= 15; reasons.append("❌ [TREND] Bearish (Harga < MA20)")

            if p > row['VWAP']: score += 10; reasons.append("✅ [CONTROL] Harga > VWAP")
            else: score -= 10; reasons.append("❌ [CONTROL] Harga < VWAP")

            if row['ADX'] > 25:
                if p > row['MA20']: score += 10; reasons.append(f"✅ [POWER] Tren Kuat (ADX {row['ADX']:.1f})")
                else: score -= 10; reasons.append(f"❌ [POWER] Jualan Kuat (ADX {row['ADX']:.1f})")
            else: reasons.append("⚠️ [POWER] Tren Lemah/Sideways")

            if row['Hist'] > 0: score += 5; reasons.append("✅ [MOMENTUM] MACD Positif")
            
            if row['StochRSI'] < 0.2: score += 10; reasons.append("✅ [TIMING] Oversold (Murah)")
            elif row['StochRSI'] > 0.8: score -= 5; reasons.append("⚠️ [TIMING] Overbought (Mahal)")
                
            if p >= res_price: score += 5; reasons.append("🚀 [BREAKOUT] Jebol Resistance Harian!")

        score = max(0, min(score, 100))

        sl_dist = row['ATR'] * 1.5
        raw_sl = p - sl_dist
        if (p - raw_sl) / p < 0.01: raw_sl = p * 0.985 
        sl_final = self.round_price(raw_sl)
        if sl_final >= p: sl_final = p - (2 if p<200 else 5)

        risk = p - sl_final
        risk_pct = (risk / p) * 100
        
        tp1 = self.round_price(p + (risk * 1.5))
        tp2 = self.round_price(p + (risk * 3.0))
        tp3 = self.round_price(p + (risk * 5.0))

        if is_zombie:
            rec_text = "⛔ SAHAM TIDUR (ZOMBIE)"
            rec_class = "box-sell"
        elif score >= 75:
            rec_text = "🚀 GAS POL / HAJAR KANAN"
            rec_class = "box-buy"
        elif score >= 60:
            rec_text = "✅ BUY ON WEAKNESS"
            rec_class = "box-buy"
        elif score >= 40:
            rec_text = "⚠️ WAIT / SPECULATIVE"
            rec_class = "box-wait"
        else:
            rec_text = "⛔ AVOID / SELL"
            rec_class = "box-sell"

        return {
            'data': row, 'score': score, 'reasons': reasons,
            'sl': sl_final, 'risk_pct': risk_pct,
            'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'rec_text': rec_text, 'rec_class': rec_class,
            'support': sup_price, 'resistance': res_price,
            'volatility': volatility_pct, 'is_zombie': is_zombie
        }

# ==========================================
# 3. INTERFACE DASHBOARD (WEB UI)
# ==========================================

with st.sidebar:
    st.title("🎛️ CONTROL PANEL")
    with st.form(key='analysis_form'):
        ticker_input = st.text_input("Kode Saham", "").upper()
        analyze_btn = st.form_submit_button("MULAI ANALISA", type="primary")
    st.divider()
    st.info("💡 **Golden Time:** 09:15 - 11:30 WIB")

if analyze_btn and ticker_input:
    bot = GodModeEngine(ticker_input)
    
    with st.spinner(f'📡 Mengakses Data: {ticker_input}...'):
        status = bot.fetch_data()

    if status == "OK":
        res = bot.analyze_market()
        d = res['data']
        p = d['Close']
        
        # --- TIMEZONE FIX (UTC SERVER TO WIB) ---
        # 1. Candle Time
        last_candle_time = d['Datetime'].to_pydatetime()
        if last_candle_time.tzinfo is not None:
            last_candle_time_utc = last_candle_time.astimezone(timezone.utc)
            last_candle_time_wib = last_candle_time_utc + timedelta(hours=7)
            last_candle_time_naive = last_candle_time_wib.replace(tzinfo=None)
        else:
            last_candle_time_naive = last_candle_time
            
        # 2. Scan Time (Server Time + 7 Hours)
        server_now = datetime.now(timezone.utc)
        wib_now = server_now + timedelta(hours=7)
        
        # 3. Delay Calculation
        # Gunakan wib_now (yang sudah timezone aware tapi naive-like) vs candle naive
        diff = wib_now.replace(tzinfo=None) - last_candle_time_naive
        delay_min = int(diff.total_seconds() / 60)
        
        if delay_min <= 5: delay_txt = f"<span class='c-green'>Realtime (<5m)</span>"
        elif delay_min <= 20: delay_txt = f"<span class='c-yellow'>Delay Wajar ({delay_min}m)</span>"
        else: delay_txt = f"<span class='c-red'>MARKET TUTUP / DATA LAMA ({int(delay_min/60)}j {delay_min%60}m)</span>"

        # --- HEADER INFO ---
        st.markdown(f"### ⚡ ANALISA SAHAM: {ticker_input}")
        c1, c2 = st.columns([2, 1])
        with c1:
            # Tampilkan WIB NOW dan beri label UTC+7
            st.markdown(f"🕒 Scan: {wib_now.strftime('%H:%M:%S')} | 🕯️ Candle: {last_candle_time_naive.strftime('%H:%M:%S')} | ⏳ Status: {delay_txt}", unsafe_allow_html=True)
        
        st.divider()

        # --- TOP SECTION: PRICE & REKOMENDASI ---
        col_main1, col_main2 = st.columns([1, 2])
        with col_main1:
            st.markdown(f"<div class='metric-card'><div class='label-font'>HARGA SAAT INI</div><div class='huge-font'>{p:.0f}</div></div>", unsafe_allow_html=True)
        with col_main2:
            st.markdown(f"<div class='{res['rec_class']}'><div class='huge-font'>{res['rec_text']}</div><div>SKOR KUALITAS: {res['score']}/100</div></div>", unsafe_allow_html=True)

        # --- SECTION: HASIL ANALISA (AUDIT) - FIXED RENDER ---
        st.markdown("<br><h5>🔍 HASIL ANALISA </h5>", unsafe_allow_html=True)
        
        # FIX: Generate HTML string lengkap dulu baru di-render
        audit_html = '<div class="audit-box">'
        for r in res['reasons']:
            # Tentukan warna teks berdasarkan keyword
            # Note: Icon sudah ada di dalam string 'r', jadi tidak perlu tambah icon lagi.
            if "[BAHAYA]" in r or "[TREND] Bearish" in r or "[CONTROL] Harga <" in r or "Jualan Kuat" in r:
                text_cls = "c-red"
            elif "[INFO]" in r or "Tren Lemah" in r or "Overbought" in r:
                text_cls = "c-yellow"
            elif "[BREAKOUT]" in r:
                text_cls = "c-magenta"
            else:
                text_cls = "c-green"
            
            # Tambahkan baris ke HTML string
            audit_html += f'<div style="margin-bottom:5px"><span class="{text_cls}">{r}</span></div>'
        
        audit_html += '</div>'
        
        # Render satu kali saja
        st.markdown(audit_html, unsafe_allow_html=True)

        # --- SECTION: SESSION INTRADAY DATA ---
        st.markdown("<br><h5>📊 DATA SESSION INTRADAY</h5>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        
        res_color = "c-magenta" if p >= res['resistance'] else "c-red"
        res_label = "🚀 BREAKOUT!" if p >= res['resistance'] else "Atap Harian"
        
        with m1: st.markdown(f"<div class='metric-card'><span class='label-font'>RESISTANCE</span><br><span class='big-font {res_color}'>{res['resistance']:.0f}</span><br><small>{res_label}</small></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-card'><span class='label-font'>SUPPORT</span><br><span class='big-font c-green'>{res['support']:.0f}</span><br><small>Lantai Harian</small></div>", unsafe_allow_html=True)
        with m3: 
             vol_cls = "c-green" if res['volatility'] >= 2.0 else "c-red"
             st.markdown(f"<div class='metric-card'><span class='label-font'>VOLATILITAS</span><br><span class='big-font {vol_cls}'>{res['volatility']:.2f}%</span><br><small>Range Harian</small></div>", unsafe_allow_html=True)

        # --- SECTION: DATA INDIKATOR ---
        st.markdown("<br><h5>📈 DATA INDIKATOR</h5>", unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns(4)
        
        vwap_cls = "c-green" if p > d['VWAP'] else "c-red"
        adx_cls = "c-green" if d['ADX'] > 25 else ("c-red" if d['ADX'] < 20 else "c-yellow")
        stoch_cls = "c-green" if d['StochRSI'] < 0.2 else ("c-red" if d['StochRSI'] > 0.8 else "c-yellow")
        
        with i1: st.markdown(f"<div class='metric-card'><small>MA 5 (Support)</small><br><span class='big-font'>{d['MA5']:.0f}</span></div>", unsafe_allow_html=True)
        with i2: st.markdown(f"<div class='metric-card'><small>VWAP (Bandar)</small><br><span class='big-font {vwap_cls}'>{d['VWAP']:.0f}</span></div>", unsafe_allow_html=True)
        with i3: st.markdown(f"<div class='metric-card'><small>ADX (Power)</small><br><span class='big-font {adx_cls}'>{d['ADX']:.1f}</span></div>", unsafe_allow_html=True)
        with i4: st.markdown(f"<div class='metric-card'><small>StochRSI</small><br><span class='big-font {stoch_cls}'>{d['StochRSI']:.2f}</span></div>", unsafe_allow_html=True)

        # --- TRADE PLAN & LOGIC ---
        if res['is_zombie']:
            st.error("⛔ ALASAN PENOLAKAN: Saham Tidur (Range Gerak < 2%). Hindari!")
        
        elif "AVOID" not in res['rec_text']:
            st.markdown("<br><h5>📋 TRADE PLAN DETAIL</h5>", unsafe_allow_html=True)
            
            p1, p2 = st.columns([1, 1])
            entry_min = bot.round_price(min(p, d['MA5']))
            entry_max = bot.round_price(max(p, d['MA5']))
            
            with p1:
                if entry_min == entry_max:
                    st.markdown(f"<div class='plan-box'><span class='label-font c-cyan'>🛒 ENTRY POINT</span><br><span class='big-font'>{entry_min:.0f}</span><br><small>HARGA TEPAT DI SUPPORT</small></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='plan-box'><span class='label-font c-cyan'>🛒 ENTRY RANGE</span><br><span class='big-font'>{entry_min:.0f} - {entry_max:.0f}</span></div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='plan-box' style='border-color:#FF4B4B'><span class='label-font c-red'>🛡️ STOP LOSS</span><br><span class='big-font'>{res['sl']:.0f}</span><br><small>Risk: -{res['risk_pct']:.1f}%</small></div>", unsafe_allow_html=True)

            with p2:
                g1 = ((res['tp1'] - p) / p) * 100
                g2 = ((res['tp2'] - p) / p) * 100
                g3 = ((res['tp3'] - p) / p) * 100
                
                st.markdown(f"""
                <div class='metric-card' style='text-align:left'>
                    <span class='c-green'>🎯 TP 1 :  </span> <b class='big-font'>{res['tp1']:.0f}</b> (+{g1:.1f}%)<br>
                    <span class='c-green'>🚀 TP 2 :  </span> <b class='big-font'>{res['tp2']:.0f}</b> (+{g2:.1f}%)<br>
                    <span class='c-magenta'>💎 GREEDY TP :  </span> <b class='big-font'>{res['tp3']:.0f}</b> (+{g3:.1f}%)
                </div>
                """, unsafe_allow_html=True)

            # VALIDASI AKHIR
            st.markdown("""
            <div style='background-color:#332b00; padding:15px; margin-top:10px; border-radius:5px; border:1px solid #FFEB3B;'>
                <b class='c-yellow'>⚠️ VALIDASI AKHIR (WAJIB CEK MANUAL):</b><br>
                1. <span class='c-cyan'>ORDER BOOK</span>: Bid Tebal? Offer Dimakan?<br>
                2. <span class='c-cyan'>BANDARMOLOGY</span>: Top Buyer = Institusi/Asing?<br>
                3. <span class='c-cyan'>RUNNING TRADE</span>: Transaksi ramai/cepat?<br>
                <b>>>> JIKA SEMUA VALID, EKSEKUSI SESUAI PLAN DI ATAS. <<<</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("⛔ SETUP TIDAK VALID. JANGAN ENTRY!!!")

    elif status == "EMPTY":
        st.error("❌ Data tidak ditemukan. Cek kode saham.")
    elif status == "FEW_DATA":
        st.error("❌ Data terlalu sedikit (Saham baru/Tidak likuid).")
    else:
        st.error(f"❌ Error System: {status}")
else:
    if not ticker_input:
        st.info("👈 Masukkan Kode Saham di Sidebar dan Tekan Enter untuk memulai.")
