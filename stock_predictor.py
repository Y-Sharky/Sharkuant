"""
智能体股价预测模块 v1.8 (支持缓存和更多指标 + CNN集成图像预测)
功能：
- 技术面分析（MA, MACD, RSI, KDJ, 布林带, 成交量均线等）
- 新闻事件影响评分（基于新闻情感和影响力，考虑时间衰减）
- CNN图像预测（基于论文(Re-)Imag(in)ing Price Trends，使用集成模型）
- 综合预测输出（涨跌趋势、支撑压力、操作建议）
- K线图绘制（K线、均线、成交量，带图例和网格）
依赖：tushare, pandas, numpy, mplfinance, matplotlib, torch, PIL
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import mplfinance as mpf
import matplotlib.pyplot as plt
import sys
import torch

# 确保能导入同级目录的 cnn_predictor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from cnn_predictor import generate_stock_image, load_ensemble_models, predict_ensemble
    CNN_AVAILABLE = True
except ImportError:
    print("警告：未找到 cnn_predictor 模块，CNN预测功能将禁用")
    CNN_AVAILABLE = False

# 全局变量（集成模型列表）
cnn_models = []
cnn_device = None
LOOKBACK_LIST = [5, 20, 60]  # 用于多尺度图像生成


def _init_cnn_model(model_paths=None):
    global cnn_models, cnn_device, CNN_AVAILABLE
    if not CNN_AVAILABLE:
        return
    try:
        if model_paths is None:
            model_paths = [
                'models/stock_cnn_pretrained_seed42.pth',
                'models/stock_cnn_pretrained_seed43.pth',
                'models/stock_cnn_pretrained_seed44.pth',
            ]
        # ========== 添加调试打印 ==========
        print(f"Attempting to load models from {model_paths}")
        for p in model_paths:
            print(f"  {p} exists: {os.path.exists(p)}")
        # ================================
        cnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn_models = load_ensemble_models(model_paths, cnn_device)
        if cnn_models:
            CNN_AVAILABLE = True
            print(f"✅ 集成CNN模型加载成功，共 {len(cnn_models)} 个模型，设备: {cnn_device}")
        else:
            print("⚠️ 未找到任何CNN模型文件，CNN预测功能禁用")
            CNN_AVAILABLE = False
    except Exception as e:
        print(f"❌ CNN模型加载失败: {e}")
        CNN_AVAILABLE = False


# ==================== 全局设置：中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ==================== 第一部分：初始化Tushare ====================
token = os.getenv('TUSHARE_TOKEN', '5594f3528180b170626430040a26acff46e86c5b8c98dc4b2f1094a5')
ts.set_token(token)
pro = ts.pro_api()

# 缓存目录
CACHE_DIR = './'
NEWS_DATA_FILE = os.path.join(CACHE_DIR, 'news_data.csv')
RECOMMENDED_STOCKS_FILE = os.path.join(CACHE_DIR, '推荐股票.csv')
DAILY_CACHE_DIR = os.path.join(CACHE_DIR, 'daily_cache')

if not os.path.exists(DAILY_CACHE_DIR):
    os.makedirs(DAILY_CACHE_DIR)


# ==================== 数据获取函数 ====================
def get_stock_basic():
    """获取A股基础信息"""
    return pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,market')


def get_daily_data(ts_code, start_date=None, end_date=None, use_cache=True):
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=200)).strftime('%Y%m%d')  # 改为200天

    cache_file = os.path.join(DAILY_CACHE_DIR, f"{ts_code}_{start_date}_{end_date}.csv")

    if use_cache and os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"使用缓存数据: {cache_file}")
            return df
        except Exception as e:
            print(f"读取缓存失败: {e}，重新获取")

    print(f"从 Tushare 获取 {ts_code} 日线数据...")
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date,
                   fields='trade_date,open,high,low,close,vol')
    if df.empty:
        print(f"警告：{ts_code} 无日线数据")
        return pd.DataFrame()

    df = df.sort_values('trade_date')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)

    if use_cache:
        try:
            df.to_csv(cache_file)
            print(f"缓存已保存: {cache_file}")
        except Exception as e:
            print(f"保存缓存失败: {e}")

    return df


# ==================== 技术指标计算 ====================
def calculate_technical_indicators(df):
    """
    计算常用技术指标并返回最新值
    """
    if df.empty or len(df) < 20:
        return None
    df = df.copy()

    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()

    df['VMA5'] = df['vol'].rolling(5).mean()
    df['VMA10'] = df['vol'].rolling(10).mean()

    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    low_9 = df['low'].rolling(9).min()
    high_9 = df['high'].rolling(9).max()
    df['RSV'] = 100 * (df['close'] - low_9) / (high_9 - low_9)
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    df['BB_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std

    latest = df.iloc[-1]
    return {
        'close': latest['close'],
        'volume': latest['vol'],
        'MA5': latest['MA5'],
        'MA10': latest['MA10'],
        'MA20': latest['MA20'],
        'VMA5': latest['VMA5'],
        'VMA10': latest['VMA10'],
        'MACD': latest['MACD'],
        'MACD_signal': latest['MACD_signal'],
        'MACD_hist': latest['MACD_hist'],
        'RSI': latest['RSI'],
        'K': latest['K'],
        'D': latest['D'],
        'J': latest['J'],
        'BB_upper': latest['BB_upper'],
        'BB_lower': latest['BB_lower'],
        'trend_short': '上升' if latest['MA5'] > latest['MA10'] else '下降',
        'trend_medium': '上升' if latest['MA10'] > latest['MA20'] else '下降',
        'volume_ratio': latest['vol'] / df['vol'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 1.0,
    }


def identify_candlestick_patterns(df):
    """
    识别常见K线形态
    """
    if df.empty or len(df) < 2:
        return []
    df = df.copy()
    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last['close'] - last['open'])
    lower_shadow = last['open'] - last['low'] if last['close'] >= last['open'] else last['close'] - last['low']
    upper_shadow = last['high'] - last['close'] if last['close'] >= last['open'] else last['high'] - last['open']

    if lower_shadow > 2 * body and upper_shadow < 0.3 * body:
        if last['close'] > last['open']:
            patterns.append('锤子线(看涨)')
        else:
            patterns.append('上吊线(看跌)')

    if body < (last['high'] - last['low']) * 0.1:
        patterns.append('十字星')

    if prev['close'] < prev['open'] and last['close'] > last['open']:
        if last['open'] < prev['close'] and last['close'] > prev['open']:
            patterns.append('看涨吞没')

    if prev['close'] > prev['open'] and last['close'] < last['open']:
        if last['open'] > prev['close'] and last['close'] < prev['open']:
            patterns.append('看跌吞没')

    return patterns


# ==================== 新闻影响分析 ====================
def load_news_data():
    if not os.path.exists(NEWS_DATA_FILE):
        print(f"警告：新闻数据文件 {NEWS_DATA_FILE} 不存在，将不考虑新闻影响。")
        return pd.DataFrame()
    df = pd.read_csv(NEWS_DATA_FILE)
    for col in ['ai_industries', 'ai_concepts', 'ai_companies']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
    if 'time_raw' in df.columns:
        try:
            df['pub_date'] = pd.to_datetime(df['time_raw'], errors='coerce')
        except:
            df['pub_date'] = None
    return df


def get_news_impact_for_stock(ts_code, news_df, half_life_days=7):
    stock_basic = get_stock_basic()
    row = stock_basic[stock_basic['ts_code'] == ts_code]
    if row.empty:
        return 0.0
    stock_name = row.iloc[0]['name']

    related_news = news_df[news_df['ai_companies'].apply(lambda x: stock_name in x if isinstance(x, list) else False)]
    if related_news.empty:
        return 0.0

    now = datetime.now()
    total_effect = 0.0
    total_weight = 0.0
    for _, row in related_news.iterrows():
        sentiment = row.get('sentiment', 0.0)
        impact = row.get('impact', 0)
        pub_date = row.get('pub_date', None)
        if pub_date is not None and pd.notna(pub_date):
            days_diff = (now - pub_date).total_seconds() / 86400
            if days_diff < 0:
                days_diff = 0
            weight = np.exp(-np.log(2) * days_diff / half_life_days)
        else:
            weight = 1.0
        total_effect += sentiment * impact * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    avg_effect = total_effect / total_weight
    return avg_effect


# ==================== 多尺度图像生成（用于CNN预测）====================
def generate_multi_scale_image(df, lookback_list=[5, 20, 60], image_size=(112, 112)):
    """
    为多通道CNN生成三通道图像（5天、20天、60天）
    df: 包含 open, high, low, close, vol 的DataFrame（索引为日期）
    lookback_list: 三个时间窗口
    image_size: 输出图像尺寸
    返回 (3, H, W) 的 numpy 数组，若数据不足则返回 None
    """
    max_lookback = max(lookback_list)
    if len(df) < max_lookback:
        return None
    imgs = []
    for lookback in lookback_list:
        window = df.iloc[-lookback:]  # 取最近 lookback 天
        img = generate_stock_image(window, image_size=image_size)
        if img is None:
            return None
        imgs.append(img)
    return np.stack(imgs, axis=0)  # (3, H, W)


# ==================== K线图绘制 ====================
def plot_kline(ts_code, df, stock_name='', save=True):
    """绘制K线图、均线、成交量，可选择保存或返回figure"""
    if df.empty or len(df) < 20:
        print(f"数据不足，无法绘制{ts_code}的K线图")
        return None
    plot_df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'vol': 'Volume'
    })
    mc = mpf.make_marketcolors(up='red', down='green', edge='inherit', wick='inherit', volume='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='lightgray', y_on_right=False)
    mav = (5, 10, 20)
    title = f"{stock_name} ({ts_code}) K线图"
    if save:
        filename = f"{ts_code}_kline.png"
        mpf.plot(plot_df, type='candle', style=s, title=title, ylabel='价格',
                 volume=True, mav=mav, savefig=filename, figsize=(12, 8))
        print(f"K线图已保存至 {filename}")
        return filename
    else:
        fig, axes = mpf.plot(plot_df, type='candle', style=s, title=title, ylabel='价格',
                             volume=True, mav=mav, returnfig=True, figsize=(12, 8))
        return fig


# ==================== 综合预测（支持外部传入CNN概率）====================
def predict_stock(ts_code, news_df, cnn_prob=None):
    """
    对单只股票进行综合预测
    参数:
        ts_code: 股票代码
        news_df: 新闻数据
        cnn_prob: 外部计算好的CNN上涨概率（0~1），若为None则跳过CNN评分
    返回字典包含预测结果
    """
    df = get_daily_data(ts_code)
    if df.empty:
        return {'股票代码': ts_code, 'error': '无日线数据'}
    indicators = calculate_technical_indicators(df)
    if indicators is None:
        return {'股票代码': ts_code, 'error': '数据不足'}
    patterns = identify_candlestick_patterns(df)
    news_impact = get_news_impact_for_stock(ts_code, news_df)

    # CNN预测 - 只使用外部传入的概率，内部不再自行预测
    if cnn_prob is not None:
        cnn_up_prob = cnn_prob
        cnn_signal = "看涨" if cnn_up_prob > 0.5 else "看跌"
    else:
        cnn_up_prob = None
        cnn_signal = "未启用"

    # 综合评分
    score = 0.0
    signals = []

    # 移动平均线信号
    if indicators['MA5'] > indicators['MA10']:
        score += 2
        signals.append('MA金叉')
    else:
        score -= 1
        signals.append('MA死叉')

    # MACD信号
    if indicators['MACD'] > indicators['MACD_signal']:
        score += 2
        signals.append('MACD金叉')
    else:
        score -= 1

    # RSI信号
    if indicators['RSI'] < 30:
        score += 3
        signals.append('RSI超卖')
    elif indicators['RSI'] > 70:
        score -= 3
        signals.append('RSI超买')
    else:
        score += 1

    # KDJ信号
    if indicators['J'] < 20:
        score += 2
        signals.append('KDJ超卖')
    elif indicators['J'] > 80:
        score -= 2
        signals.append('KDJ超买')

    # 成交量信号
    if indicators['volume_ratio'] > 1.5:
        score += 1
        signals.append('放量')
    elif indicators['volume_ratio'] < 0.5:
        score -= 1
        signals.append('缩量')

    # 布林带信号
    if indicators['close'] > indicators['BB_upper']:
        score -= 2
        signals.append('突破上轨(超买)')
    elif indicators['close'] < indicators['BB_lower']:
        score += 2
        signals.append('跌破下轨(超卖)')

    # 新闻影响
    if news_impact > 2:
        score += 3
        signals.append('重大利好')
    elif news_impact > 0:
        score += 1
        signals.append('小幅利好')
    elif news_impact < -2:
        score -= 3
        signals.append('重大利空')
    elif news_impact < 0:
        score -= 1
        signals.append('小幅利空')

    # K线形态
    if '看涨吞没' in patterns or '锤子线(看涨)' in patterns:
        score += 2
        signals.append('看涨形态')
    elif '看跌吞没' in patterns or '上吊线(看跌)' in patterns:
        score -= 2
        signals.append('看跌形态')

    # CNN结果融入综合评分
    if cnn_up_prob is not None:
        if cnn_up_prob > 0.7:
            score += 3
            signals.append('CNN强烈看涨')
        elif cnn_up_prob > 0.55:
            score += 2
            signals.append('CNN看涨')
        elif cnn_up_prob < 0.3:
            score -= 3
            signals.append('CNN强烈看跌')
        elif cnn_up_prob < 0.45:
            score -= 2
            signals.append('CNN看跌')

    # 趋势判断
    trend_signal = "震荡"
    if score >= 5:
        trend_signal = "强烈看涨"
    elif score >= 2:
        trend_signal = "看涨"
    elif score <= -5:
        trend_signal = "强烈看跌"
    elif score <= -2:
        trend_signal = "看跌"

    # 目标价预测
    current_price = indicators['close']
    volatility = df['close'].pct_change().std() * np.sqrt(5)
    if trend_signal in ['强烈看涨', '看涨']:
        target_up = current_price * (1 + volatility * 2)
        target_down = current_price * (1 - volatility * 0.5)
    elif trend_signal in ['强烈看跌', '看跌']:
        target_up = current_price * (1 + volatility * 0.5)
        target_down = current_price * (1 - volatility * 2)
    else:
        target_up = current_price * (1 + volatility)
        target_down = current_price * (1 - volatility)

    stock_basic = get_stock_basic()
    name_row = stock_basic[stock_basic['ts_code'] == ts_code]
    stock_name = name_row['name'].values[0] if not name_row.empty else ts_code

    return {
        '股票代码': ts_code,
        '股票名称': stock_name,
        '当前价格': round(current_price, 2),
        '预测趋势': trend_signal,
        '综合得分': score,
        '支撑位': round(target_down, 2),
        '压力位': round(target_up, 2),
        '技术信号': ', '.join(signals),
        '新闻影响': round(news_impact, 2),
        'K线形态': ', '.join(patterns) if patterns else '无',
        'CNN上涨概率': round(cnn_up_prob, 3) if cnn_up_prob is not None else 'N/A',
        'CNN信号': cnn_signal
    }


# ==================== 可调用的预测函数 ====================
def save_dataframe_safely(df, base_filename, max_attempts=10):
    for i in range(max_attempts):
        if i == 0:
            filename = base_filename
        else:
            name, ext = os.path.splitext(base_filename)
            filename = f"{name}_{i}{ext}"
        try:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"结果已保存到 {filename}")
            return filename
        except PermissionError:
            if i == max_attempts - 1:
                print(f"无法保存文件，请关闭占用 {base_filename} 的程序后重试。")
                return None
            continue
    return None


def run_prediction(stock_list=None):
    """
    运行预测流程（支持CNN集成模型预测）
    """
    print("=" * 50)
    print("智能体股价预测模块 v1.8 (支持CNN集成模型)")
    print("=" * 50)

    news_df = load_news_data()
    print(f"已加载新闻数据 {len(news_df)} 条")

    # 确定股票列表
    if stock_list is None:
        if os.path.exists(RECOMMENDED_STOCKS_FILE):
            rec_df = pd.read_csv(RECOMMENDED_STOCKS_FILE)
            if '股票代码' in rec_df.columns:
                stock_list = rec_df['股票代码'].tolist()
                print(f"从 {RECOMMENDED_STOCKS_FILE} 加载 {len(stock_list)} 只推荐股票")
            else:
                stock_list = []
        else:
            stock_list = []

    if not stock_list:
        print("未指定股票列表，请输入要预测的股票代码（多个用空格分隔）：")
        user_input = input().strip()
        if user_input:
            stock_list = user_input.split()
        else:
            stock_list = ['000001.SZ', '600519.SH', '300750.SZ']

    results = []
    for ts_code in stock_list:
        print(f"\n--- 正在处理 {ts_code} ---")
        df = get_daily_data(ts_code)
        if df.empty or len(df) < 20:
            results.append({'股票代码': ts_code, 'error': '数据不足'})
            continue

        # 绘制K线图（可选，保留）
        stock_basic = get_stock_basic()
        name_row = stock_basic[stock_basic['ts_code'] == ts_code]
        stock_name = name_row['name'].values[0] if not name_row.empty else ts_code
        plot_kline(ts_code, df, stock_name)

        # CNN预测（使用集成模型）
        cnn_prob = None
        print(
            f"DEBUG: CNN_AVAILABLE={CNN_AVAILABLE}, cnn_models={len(cnn_models)}, df_len={len(df)}, max_lookback={max(LOOKBACK_LIST)}")
        if CNN_AVAILABLE and cnn_models and len(df) >= max(LOOKBACK_LIST):
            print("DEBUG: 条件满足，开始生成图像...")
            try:
                multi_img = generate_multi_scale_image(df, lookback_list=[5, 20, 60], image_size=(112, 112))
                if multi_img is not None:
                    print(f"DEBUG: 图像生成成功，shape={multi_img.shape}")
                    cnn_prob = predict_ensemble(cnn_models, cnn_device, multi_img)
                    print(f"CNN集成预测上涨概率: {cnn_prob:.4f}")
                else:
                    print("DEBUG: generate_multi_scale_image 返回 None")
            except Exception as e:
                print(f"CNN预测失败: {e}")
        else:
            print("DEBUG: 条件不满足，跳过CNN预测")

        # 综合预测
        result = predict_stock(ts_code, news_df, cnn_prob=cnn_prob)
        results.append(result)

    # 输出结果
    result_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("【股价预测结果】")
    print("=" * 50)
    print(result_df.to_string(index=False))

    output_file = '股价预测.csv'
    save_dataframe_safely(result_df, output_file)
    return result_df


# 初始化CNN模型（在所有函数定义之后）
_init_cnn_model()

if __name__ == "__main__":
    run_prediction()