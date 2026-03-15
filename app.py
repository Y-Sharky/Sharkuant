"""
智能体选股预测综合网站 (Streamlit版)
功能：
- 展示行业/概念热度排名
- 展示推荐股票列表
- 股票搜索与预测（K线图、技术指标、新闻影响）
- 手动触发数据更新（运行 stock_suggestion.py）
依赖：stock_predictor.py, stock_suggestion.py, news_data.csv, 推荐股票.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入预测模块的函数
from stock_predictor import get_stock_basic, get_daily_data, plot_kline, predict_stock, load_news_data

# ==================== 页面配置 ====================
st.set_page_config(page_title="智能体选股预测系统", layout="wide")
st.title("📊 智能体选股预测系统")

# ==================== 侧边栏：全局功能 ====================
st.sidebar.header("控制面板")

# 更新数据按钮（手动触发）
if st.sidebar.button("🔄 更新新闻与推荐数据"):
    with st.spinner("正在运行选股脚本，请稍候..."):
        try:
            # 调用 stock_suggestion.py（需确保当前目录下有该文件）
            result = subprocess.run([sys.executable, "stock_suggestion.py", "--auto"], capture_output=True, text=True)
            if result.returncode == 0:
                st.sidebar.success("数据更新成功！")
            else:
                st.sidebar.error(f"数据更新失败：{result.stderr}")
        except Exception as e:
            st.sidebar.error(f"执行出错：{e}")

st.sidebar.markdown("---")

# 股票搜索区域（预测功能）
st.sidebar.header("股票搜索")
keyword = st.sidebar.text_input("输入股票名称或代码")

if st.sidebar.button("搜索"):
    stock_basic = get_stock_basic()
    mask = (stock_basic['name'].str.contains(keyword, na=False) |
            stock_basic['symbol'].str.contains(keyword, na=False) |
            stock_basic['ts_code'].str.contains(keyword, na=False))
    result = stock_basic[mask]
    if result.empty:
        st.sidebar.warning("未找到匹配的股票")
    else:
        st.sidebar.success(f"找到 {len(result)} 只股票")
        st.session_state['search_result'] = result

if 'search_result' in st.session_state:
    result = st.session_state['search_result']
    options = result.apply(lambda x: f"{x['name']} ({x['ts_code']})", axis=1).tolist()
    selected = st.sidebar.selectbox("请选择股票", options)
    ts_code = selected.split('(')[-1].strip(')')
    stock_name = selected.split(' (')[0]
    st.session_state['selected_code'] = ts_code
    st.session_state['selected_name'] = stock_name

# ==================== 主区域：选项卡 ====================
tab1, tab2, tab3 = st.tabs(["📈 行业/概念热度", "⭐ 推荐股票", "🔍 股票预测详情"])

# ---------- 行业/概念热度 ----------
with tab1:
    col1, col2 = st.columns(2)
    # 读取行业热度
    if os.path.exists('news_data.csv'):
        df_news = pd.read_csv('news_data.csv')
        # 这里需要从 news_data.csv 中计算行业和概念热度，但实际热度是在 run_analysis_flow 中计算的，并存储在变量中，没有保存到文件。
        # 为了简化，我们可以每次运行时重新计算热度（从 news_data.csv），但需要导入 calculate_heat 函数。
        # 或者我们可以在 stock_suggestion.py 运行时将行业热度和概念热度也保存为 CSV。
        # 为了快速实现，我们先从 stock_suggestion.py 中导入 calculate_heat 函数来计算。
        try:
            from stock_suggestion import calculate_heat
            news_df = load_news_data()  # 加载带AI分析结果的新闻数据
            industry_heat, concept_heat = calculate_heat(news_df)
            with col1:
                st.subheader("行业热度排名（前20）")
                st.dataframe(industry_heat.head(20))
            with col2:
                st.subheader("概念热度排名（前20）")
                st.dataframe(concept_heat.head(20))
        except Exception as e:
            st.error(f"无法计算热度：{e}")
    else:
        st.info("暂无新闻数据，请先点击侧边栏更新数据。")

# ---------- 推荐股票 ----------
with tab2:
    if os.path.exists('推荐股票.csv'):
        rec_df = pd.read_csv('推荐股票.csv')
        st.subheader("最新推荐股票 Top 10")
        # 添加一个操作列，点击“预测”可以跳转到预测详情
        if st.button("预测选中股票", key="predict_rec"):
            # 这里简单处理：将选中的股票代码存入 session_state
            pass  # 下面用 selectbox 选择
        # 提供一个选择框，用户可以选择要预测的推荐股票
        options = rec_df.apply(lambda x: f"{x['股票名称']} ({x['股票代码']})", axis=1).tolist()
        selected_rec = st.selectbox("选择股票进行预测", options)
        if selected_rec:
            ts_code = selected_rec.split('(')[-1].strip(')')
            name = selected_rec.split(' (')[0]
            st.session_state['selected_code'] = ts_code
            st.session_state['selected_name'] = name
            st.success(f"已选择 {name}，请切换到“股票预测详情”选项卡查看。")
        st.dataframe(rec_df)
    else:
        st.info("暂无推荐股票数据，请先更新数据。")

# ---------- 股票预测详情 ----------
with tab3:
    if 'selected_code' in st.session_state:
        ts_code = st.session_state['selected_code']
        name = st.session_state['selected_name']

        col1, col2 = st.columns([2, 1])

        df = get_daily_data(ts_code)

        with col1:
            st.subheader(f"{name} ({ts_code}) K线图")
            if df is not None and len(df) >= 20:
                fig = plot_kline(ts_code, df, stock_name=name, save=False)
                st.pyplot(fig)
            else:
                st.warning("数据不足，无法绘制K线图")

        with col2:
            st.subheader("预测结果")
            news_df = load_news_data()
            if df is not None and len(df) >= 20:
                result = predict_stock(ts_code, news_df)
                st.metric("当前价格", f"{result['当前价格']} 元")
                st.metric("预测趋势", result['预测趋势'])
                st.metric("综合得分", result['综合得分'])
                st.metric("支撑位", f"{result['支撑位']} 元")
                st.metric("压力位", f"{result['压力位']} 元")
                with st.expander("详细信号"):
                    st.write(f"**技术信号**: {result['技术信号']}")
                    st.write(f"**新闻影响**: {result['新闻影响']}")
                    st.write(f"**K线形态**: {result['K线形态']}")
            else:
                st.info("暂无足够数据进行预测")

        if df is not None:
            with st.expander("查看历史数据"):
                st.dataframe(df.tail(20))
    else:
        st.info("请在侧边栏搜索股票或在推荐股票选项卡中选择股票。")