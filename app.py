"""
智能体选股预测综合网站 (Streamlit版)
功能：
- 展示行业/概念热度排名（从预先计算的 CSV 读取）
- 展示推荐股票列表
- 股票搜索与预测（K线图、技术指标、新闻影响）
- 自定义行业/概念推荐（输入关键词动态生成）
- 数据由 GitHub Actions 自动更新
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入预测模块的函数
from stock_predictor import get_stock_basic, get_daily_data, plot_kline, predict_stock, load_news_data
# 导入选股模块的函数（用于自定义推荐）
from stock_suggestion import rank_stocks, load_daily_basic, get_stock_fundamental, get_stock_technical

# ==================== 页面配置 ====================
st.set_page_config(page_title="智能体选股预测系统", layout="wide")
st.title("📊 智能体选股预测系统")

# ==================== 初始化会话状态 ====================
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0  # 默认显示行业/概念热度
if 'selected_code' not in st.session_state:
    st.session_state.selected_code = None
if 'selected_name' not in st.session_state:
    st.session_state.selected_name = None

# ==================== 侧边栏 ====================
st.sidebar.header("控制面板")

# 显示数据更新时间（如果存在缓存文件）
if os.path.exists('news_data.csv'):
    mtime_utc = datetime.fromtimestamp(os.path.getmtime('news_data.csv'))
    mtime_local = mtime_utc + timedelta(hours=8)  # 转换为北京时间
    st.sidebar.info(f"📅 数据最后更新：{mtime_local.strftime('%Y-%m-%d %H:%M')} (北京时间)")
else:
    st.sidebar.warning("暂无数据，请等待 GitHub Actions 首次运行。")

st.sidebar.markdown("🔁 [手动触发数据更新](https://github.com/Y-Sharky/Sharkuant/actions/workflows/update_data.yml)")
st.sidebar.caption("点击链接前往 GitHub Actions 手动运行更新工作流（需登录 GitHub）。")
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
    if selected:
        ts_code = selected.split('(')[-1].strip(')')
        stock_name = selected.split(' (')[0]
        # 如果选择的是新股票，则保存并跳转到预测详情
        if st.session_state.selected_code != ts_code:
            st.session_state.selected_code = ts_code
            st.session_state.selected_name = stock_name
            st.session_state.current_tab = 2  # 切换到预测详情
            st.rerun()

# ==================== 自定义选项卡 ====================
tab_options = ["📈 行业/概念热度", "⭐ 推荐股票", "🔍 股票预测详情", "⚙️ 自定义推荐"]
cols = st.columns(len(tab_options))
for i, option in enumerate(tab_options):
    if cols[i].button(option, use_container_width=True):
        st.session_state.current_tab = i
        st.rerun()

st.markdown("---")

# ==================== 根据当前选项卡显示内容 ====================
# ---------- 行业/概念热度 ----------
if st.session_state.current_tab == 0:
    st.subheader("行业/概念热度排名")
    col1, col2 = st.columns(2)
    if os.path.exists('industry_heat.csv') and os.path.exists('concept_heat.csv'):
        industry_heat = pd.read_csv('industry_heat.csv')
        concept_heat = pd.read_csv('concept_heat.csv')
        with col1:
            st.markdown("**行业热度排名（前20）**")
            st.dataframe(industry_heat.head(20))
        with col2:
            st.markdown("**概念热度排名（前20）**")
            st.dataframe(concept_heat.head(20))
    else:
        st.info("暂无热度数据，请等待 GitHub Actions 首次运行。")

# ---------- 推荐股票 ----------
elif st.session_state.current_tab == 1:
    st.subheader("最新推荐股票")
    if os.path.exists('推荐股票.csv'):
        rec_df = pd.read_csv('推荐股票.csv')
        st.markdown("**Top 10 推荐股票**")
        options = rec_df.apply(lambda x: f"{x['股票名称']} ({x['股票代码']})", axis=1).tolist()
        selected_rec = st.selectbox("选择股票进行预测", options)
        if st.button("预测选中股票", key="predict_rec"):
            ts_code = selected_rec.split('(')[-1].strip(')')
            name = selected_rec.split(' (')[0]
            st.session_state.selected_code = ts_code
            st.session_state.selected_name = name
            st.session_state.current_tab = 2
            st.rerun()
        st.dataframe(rec_df)
    else:
        st.info("暂无推荐股票数据，请等待 GitHub Actions 首次运行。")

# ---------- 股票预测详情 ----------
elif st.session_state.current_tab == 2:
    if st.session_state.selected_code is not None:
        ts_code = st.session_state.selected_code
        name = st.session_state.selected_name
        st.subheader(f"{name} ({ts_code}) 预测详情")

        col1, col2 = st.columns([2, 1])

        # 获取日线数据
        with st.spinner("正在获取数据..."):
            try:
                df = get_daily_data(ts_code)
            except Exception as e:
                st.error(f"获取数据失败：{e}")
                df = None

        with col1:
            if df is not None and len(df) >= 20:
                fig = plot_kline(ts_code, df, stock_name=name, save=False)
                st.pyplot(fig)
            elif df is not None and len(df) < 20:
                st.warning(f"数据不足（仅 {len(df)} 条），至少需要20条数据才能绘制K线图。")
            else:
                st.warning("无法获取日线数据，请检查股票代码或网络连接。")

        with col2:
            st.markdown("**预测结果**")
            news_df = load_news_data()
            if df is not None and len(df) >= 20:
                try:
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
                except Exception as e:
                    st.error(f"预测失败：{e}")
            else:
                st.info("暂无足够数据进行预测")

        if df is not None:
            with st.expander("查看历史数据"):
                st.dataframe(df.tail(20))
    else:
        st.info("请在侧边栏搜索股票或在推荐股票选项卡中选择股票。")

# ---------- 自定义推荐 ----------
elif st.session_state.current_tab == 3:
    st.subheader("自定义行业/概念推荐")
    st.markdown("请输入您关注的行业和概念关键词（可多个，用空格或逗号分隔），点击按钮重新生成推荐。")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        ind_input = st.text_input("行业关键词", help="例如：银行 证券 软件")
    with col_in2:
        con_input = st.text_input("概念关键词", help="例如：人工智能 5G 芯片")

    if st.button("生成推荐", type="primary"):
        # 解析关键词
        industries = None
        concepts = None
        if ind_input:
            import re
            industries = [k.strip() for k in re.split(r'[ ,，]+', ind_input) if k.strip()]
        if con_input:
            import re
            concepts = [k.strip() for k in re.split(r'[ ,，]+', con_input) if k.strip()]

        # 检查热度文件是否存在
        if not os.path.exists('industry_heat.csv') or not os.path.exists('concept_heat.csv'):
            st.error("热度数据文件不存在，请等待 GitHub Actions 首次运行生成数据。")
        else:
            with st.spinner("正在计算推荐，请稍候..."):
                try:
                    # 读取热度数据
                    industry_heat = pd.read_csv('industry_heat.csv')
                    concept_heat = pd.read_csv('concept_heat.csv')

                    # 调用 rank_stocks 生成推荐
                    result = rank_stocks(industry_heat, concept_heat,
                                         filter_industries=industries,
                                         filter_concepts=concepts,
                                         top_n=10)

                    if not result.empty:
                        st.success("推荐生成成功！")
                        st.dataframe(result)
                        # 保存到 session_state 供预测使用
                        st.session_state['custom_result'] = result
                    else:
                        st.warning("没有找到符合条件的股票，请尝试放宽关键词。")
                except Exception as e:
                    st.error(f"生成推荐时出错：{e}")

    # 如果之前有生成结果，可以提供一个按钮将选中的股票用于预测
    if 'custom_result' in st.session_state and not st.session_state['custom_result'].empty:
        st.markdown("---")
        st.subheader("从推荐结果中选择股票进行预测")
        options = st.session_state['custom_result'].apply(
            lambda x: f"{x['股票名称']} ({x['股票代码']})", axis=1).tolist()
        selected_custom = st.selectbox("选择股票", options, key="custom_select")
        if st.button("预测选中股票", key="predict_custom"):
            ts_code = selected_custom.split('(')[-1].strip(')')
            name = selected_custom.split(' (')[0]
            st.session_state.selected_code = ts_code
            st.session_state.selected_name = name
            st.session_state.current_tab = 2
            st.rerun()