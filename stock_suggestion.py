"""
智能体选股原型 v3.5 (交互式菜单版，集成预测，使用AkShare概念映射 + 缓存)
功能：双源新闻爬取 + AI分析 + 行业/概念热度计算 + 股票综合排名 + 用户指定行业/概念筛选 + 集成股价预测
"""

import tushare as ts
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
import json
import os
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright
import akshare as ak
from stock_predictor import predict_stock, load_news_data, plot_kline, get_daily_data  # 导入预测函数

# 检测是否在 Streamlit Cloud 或 GitHub Actions 环境
if os.getenv('STREAMLIT_RUNTIME') or os.getenv('GITHUB_ACTIONS'):
    # 在云端只定义函数，不执行任何初始化
    pass



# ==================== 第一部分：初始化Tushare ====================
ts.set_token('5594f3528180b170626430040a26acff46e86c5b8c98dc4b2f1094a5')
pro = ts.pro_api()

# 缓存股票基本信息，避免重复请求
_STOCK_BASIC_CACHE = None

def get_stock_basic(use_cache=True):
    global _STOCK_BASIC_CACHE
    if use_cache and _STOCK_BASIC_CACHE is not None:
        return _STOCK_BASIC_CACHE
    data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,market')
    _STOCK_BASIC_CACHE = data
    return data

# ==================== 已爬取新闻管理 ====================
CRAWLED_TITLES_FILE = 'crawled_titles.txt'

def load_crawled_titles():
    """从文件加载已爬取的新闻标题，返回 set"""
    if not os.path.exists(CRAWLED_TITLES_FILE):
        return set()
    with open(CRAWLED_TITLES_FILE, 'r', encoding='utf-8') as f:
        titles = {line.strip() for line in f if line.strip()}
    print(f"已加载 {len(titles)} 条已爬取新闻标题")
    return titles

def save_crawled_titles(titles, append=True):
    """将新爬取的标题追加到文件（去重），或覆盖写入"""
    if append:
        existing = set()
        if os.path.exists(CRAWLED_TITLES_FILE):
            with open(CRAWLED_TITLES_FILE, 'r', encoding='utf-8') as f:
                existing = {line.strip() for line in f}
        all_titles = existing.union({str(t) for t in titles if pd.notna(t)})
    else:
        # 覆盖模式：直接使用 titles 作为全部
        all_titles = {str(t) for t in titles if pd.notna(t)}
    with open(CRAWLED_TITLES_FILE, 'w', encoding='utf-8') as f:
        for title in all_titles:
            f.write(title + '\n')
    print(f"已保存 {len(titles)} 条新标题，累计 {len(all_titles)} 条")

# ==================== 加载行业和概念数据 ====================
print("加载行业和概念数据...")
industry_df = pd.read_excel('行业.xlsx', sheet_name='Sheet1', skiprows=1)
industry_df = industry_df[['行业名称', '行业代码']].dropna()
INDUSTRY_LIST = industry_df['行业名称'].tolist()
INDUSTRY_CODE_MAP = dict(zip(industry_df['行业名称'], industry_df['行业代码']))

concept_df = pd.read_excel('概念大全.xlsx', sheet_name='Sheet1', header=None)
concept_str = concept_df.iloc[0, 0]
CONCEPT_LIST = concept_str.split()
print(f"加载完成：行业 {len(INDUSTRY_LIST)} 个，概念 {len(CONCEPT_LIST)} 个")

# ==================== 从 AkShare 加载概念-股票映射（带缓存）====================
CONCEPT_STOCK_MAP = {}          # 概念名称 -> set(股票代码)
USE_PRECISE_CONCEPT = False     # 是否使用精确概念映射
CONCEPT_CACHE_FILE = 'concept_map_cache.json'

def load_concept_map_from_akshare():
    """尝试从 AkShare 获取概念板块成分股，构建概念-股票映射，并使用本地缓存"""
    global CONCEPT_STOCK_MAP, USE_PRECISE_CONCEPT

    # 如果存在缓存文件，直接读取
    if os.path.exists(CONCEPT_CACHE_FILE):
        try:
            with open(CONCEPT_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            # 将缓存中的列表转回 set
            CONCEPT_STOCK_MAP = {k: set(v) for k, v in cache_data.items()}
            USE_PRECISE_CONCEPT = True
            print(f"已从缓存文件加载 {len(CONCEPT_STOCK_MAP)} 个概念的成分股数据。")
            return
        except Exception as e:
            print(f"读取缓存文件失败: {e}，将重新从网络加载。")

    try:
        # 1. 获取概念列表
        print("正在从 AkShare 获取概念列表...")
        concept_df = ak.stock_board_concept_name_em()
        if concept_df is None or concept_df.empty:
            print("警告：无法获取概念列表，将使用股票名称模糊匹配进行概念筛选。")
            USE_PRECISE_CONCEPT = False
            return

        total_concepts = len(concept_df)
        print(f"共获取到 {total_concepts} 个概念，开始加载成分股（此过程可能需要几分钟，请耐心等待）...")

        # 2. 遍历每个概念，获取成分股
        success_count = 0
        for idx, (_, row) in enumerate(concept_df.iterrows()):
            concept_name = row['板块名称']
            concept_code = row['板块代码']
            # 显示进度
            if (idx + 1) % 10 == 0 or idx == 0 or idx == total_concepts - 1:
                print(f"进度: {idx+1}/{total_concepts} - 正在处理概念: {concept_name}")

            try:
                # 获取概念成分股，设置超时
                stocks_df = ak.stock_board_concept_cons_em(symbol=concept_code)
                if stocks_df is not None and not stocks_df.empty:
                    # 提取股票代码，转换为 Tushare 格式（如 000001.SZ）
                    stock_codes = set()
                    for _, srow in stocks_df.iterrows():
                        code = srow['代码']
                        # 补全后缀：深市 .SZ，沪市 .SH
                        if code.startswith('6'):
                            ts_code = code + '.SH'
                        else:
                            ts_code = code + '.SZ'
                        stock_codes.add(ts_code)
                    if stock_codes:
                        CONCEPT_STOCK_MAP[concept_name] = stock_codes
                        success_count += 1
            except Exception as e:
                print(f"获取概念 {concept_name} 成分股失败: {e}")
                continue
            time.sleep(0.2)  # 礼貌爬取，避免被封

        if CONCEPT_STOCK_MAP:
            USE_PRECISE_CONCEPT = True
            print(f"成功加载 {success_count}/{total_concepts} 个概念的成分股数据。")
            # 保存到缓存文件
            try:
                # 将 set 转 list 以便 JSON 序列化
                cache_data = {k: list(v) for k, v in CONCEPT_STOCK_MAP.items()}
                with open(CONCEPT_CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                print(f"概念映射已缓存到 {CONCEPT_CACHE_FILE}")
            except Exception as e:
                print(f"保存缓存文件失败: {e}")
        else:
            USE_PRECISE_CONCEPT = False
            print("警告：未加载到任何概念数据，将使用股票名称模糊匹配进行概念筛选。")
    except Exception as e:
        print(f"加载概念映射失败: {e}，将使用股票名称模糊匹配进行概念筛选。")
        USE_PRECISE_CONCEPT = False



# ==================== 全局可调参数 ====================
NEWS_WEIGHT_FACTOR = 10      # 新闻提及权重系数（当前未使用）
SENTIMENT_WEIGHT = 10        # 情感得分系数（当前未使用）
IMPACT_WEIGHT = 2            # 影响力系数（当前未使用）
HALF_LIFE_DAYS = 7           # 时间衰减半衰期（天）

# ==================== AI智能体配置 ====================
AI_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
AI_API_KEY = "sk-e0caf789280d422faa9cd1758496e8ec"  # 请替换为您的有效API Key
AI_MODEL = "qwen-turbo"

# ==================== 第二部分：新闻数据获取 ====================

def crawl_eastmoney_with_playwright(page_num_limit=10, existing_titles=None):
    """增量爬取东方财富网新闻：从第一页开始，直到遇到已经存在的标题或爬满 page_num_limit 页。"""
    if existing_titles is None:
        existing_titles = set()
    all_news = []
    stop_flag = False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for page_index in range(1, page_num_limit + 1):
            if stop_flag:
                break
            url = f"https://finance.eastmoney.com/a/cgnjj_{page_index}.html"
            print(f"正在处理: {url}")
            try:
                page.goto(url)
                page.wait_for_selector('ul#newsListContent li', timeout=10000)
                items = page.query_selector_all('ul#newsListContent li')
                print(f"第{page_index}页找到 {len(items)} 条新闻")
                for item in items:
                    title_elem = item.query_selector('p.title a')
                    if not title_elem:
                        continue
                    title = title_elem.inner_text().strip()
                    if title in existing_titles:
                        print(f"遇到已爬取新闻: {title}，停止爬取")
                        stop_flag = True
                        break
                    link = title_elem.get_attribute('href')
                    time_elem = item.query_selector('p.time')
                    pub_time = time_elem.inner_text().strip() if time_elem else ''
                    if title and link:
                        all_news.append({
                            'title': title,
                            'link': link,
                            'source': 'eastmoney',
                            'time_raw': pub_time,
                            'content': ''
                        })
                if stop_flag:
                    break
                time.sleep(1)
            except Exception as e:
                print(f"处理第{page_index}页时出错: {e}")
                continue
        browser.close()
    print(f"本次新爬取东方财富新闻 {len(all_news)} 条")
    return pd.DataFrame(all_news)

def get_akshare_news(days=2):
    """使用 AKShare 获取财经新闻（只保留重要性1或2的事件）"""
    all_news = []
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y%m%d')
            try:
                df_cal = ak.news_economic_baidu(date=date_str)
                if df_cal is not None and not df_cal.empty:
                    for _, row in df_cal.iterrows():
                        if row.get('重要性') in [1, 2]:
                            title = f"{row['地区']} {row['事件']}"
                            time_raw = f"{row['日期']} {row['时间']}"
                            content = f"公布值: {row['公布']}, 预期: {row['预期']}, 前值: {row['前值']}"
                            all_news.append({
                                'title': title,
                                'link': '',
                                'source': 'baidu_calendar',
                                'time_raw': time_raw,
                                'content': content
                            })
            except Exception as e:
                print(f"获取百度日历 {date_str} 失败: {e}")
            current += timedelta(days=1)
            time.sleep(0.5)

        try:
            df_em = ak.stock_news_em(symbol="")
            if df_em is not None and not df_em.empty:
                col_map = {'新闻标题': 'title', '新闻链接': 'link', '发布时间': 'time_raw', '新闻内容': 'content'}
                df_em = df_em.rename(columns=col_map)
                for _, row in df_em.iterrows():
                    all_news.append({
                        'title': row.get('title', ''),
                        'link': row.get('link', ''),
                        'source': 'eastmoney_em',
                        'time_raw': row.get('time_raw', ''),
                        'content': row.get('content', '')
                    })
        except Exception as e:
            print(f"东方财富新闻获取失败（可忽略）: {e}")

        try:
            df_sina = ak.news_sina()
            if df_sina is not None and not df_sina.empty:
                if 'title' in df_sina.columns and 'time' in df_sina.columns:
                    df_sina = df_sina.rename(columns={'time': 'time_raw'})
                    for _, row in df_sina.iterrows():
                        all_news.append({
                            'title': row.get('title', ''),
                            'link': '',
                            'source': 'sina',
                            'time_raw': row.get('time_raw', ''),
                            'content': row.get('content', '')
                        })
        except Exception as e:
            print(f"新浪财经新闻获取失败（可忽略）: {e}")

        print(f"AKShare 新闻获取成功，共 {len(all_news)} 条")
        return pd.DataFrame(all_news)
    except Exception as e:
        print(f"AKShare 新闻获取整体异常: {e}")
        return pd.DataFrame()

def get_akshare_news_deduplicated(days=2, existing_titles=None):
    """获取 AKShare 新闻，并过滤掉已经存在的标题"""
    if existing_titles is None:
        existing_titles = set()
    df = get_akshare_news(days)
    if df.empty:
        return df
    mask = ~df['title'].isin(existing_titles)
    new_df = df[mask].copy()
    print(f"AKShare 新闻总数 {len(df)}，新增 {len(new_df)} 条")
    return new_df

def fetch_news_content(url):
    """从东方财富新闻链接抓取正文"""
    if not url or 'eastmoney.com' not in url:
        return ''
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')
        for selector in ['div.articleBody', 'div.content', 'div.article-content', 'div.text']:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text(strip=True)[:2000]
        paragraphs = soup.find_all('p')
        return '\n'.join([p.get_text(strip=True) for p in paragraphs])[:2000]
    except Exception as e:
        print(f"抓取正文失败: {url}, 错误: {e}")
        return ''

def parse_news_time(time_str):
    """将多种时间格式转换为datetime对象"""
    if not time_str or not isinstance(time_str, str):
        return None
    time_str = time_str.strip()
    pattern_year = r'(\d{4})年(\d{2})月(\d{2})日 (\d{2}):(\d{2})'
    match = re.match(pattern_year, time_str)
    if match:
        year, month, day, hour, minute = match.groups()
        dt_str = f"{year}-{month}-{day} {hour}:{minute}:00"
        try:
            return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        except:
            pass
    pattern_no_year = r'(\d{2})月(\d{2})日 (\d{2}):(\d{2})'
    match = re.match(pattern_no_year, time_str)
    if match:
        month, day, hour, minute = match.groups()
        current_year = datetime.now().year
        dt_str = f"{current_year}-{month}-{day} {hour}:{minute}:00"
        try:
            return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        except:
            pass
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except:
        pass
    try:
        return datetime.strptime(time_str, '%Y-%m-%d')
    except:
        pass
    return None

def time_decay_weight(pub_time, half_life_days=HALF_LIFE_DAYS):
    if pub_time is None:
        return 0.0
    now = datetime.now()
    days_diff = (now - pub_time).total_seconds() / 86400
    if days_diff < 0:
        days_diff = 0
    decay = np.exp(-np.log(2) * days_diff / half_life_days)
    return decay

# ==================== AI智能体分析 ====================
def analyze_news_with_ai(title, content=''):
    """调用AI智能体分析新闻，返回结构化结果"""
    prompt = f"""请分析以下财经新闻，返回JSON格式结果。

标题：{title}
正文：{content if content else '无'}

要求：
- news_type: 新闻类型，只能是以下之一：'政策'、'行业'、'公司'、'其他'。
- sentiment: 情感得分，范围-1到1，-1代表极大利空，0中性，1极大利好。
- industries: 涉及行业列表，请从以下行业列表中选择最匹配的行业名称（可多个，若无则[]）：
{INDUSTRY_LIST}
- concepts: 涉及概念列表，请从以下概念列表中选择最匹配的概念名称（可多个，若无则[]）：
{CONCEPT_LIST}
- impact: 对行业/概念的影响程度，0~10，0无影响，10极大影响。
- companies: 明确提及的公司名称列表（如新闻中直接提到"宁德时代"则写全称，若无则[]）。

请只输出JSON，不要其他解释。"""
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": AI_MODEL,
        "input": {"messages": [{"role": "user", "content": prompt}]}
    }
    default = {
        'news_type': '其他',
        'sentiment': 0.0,
        'industries': [],
        'concepts': [],
        'impact': 0,
        'companies': []
    }
    try:
        resp = requests.post(AI_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if 'output' in data and 'text' in data['output']:
            content = data['output']['text']
        else:
            print("未知的API返回格式")
            return default
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        result = json.loads(content)
        for key in default:
            if key not in result:
                result[key] = default[key]
        return result
    except Exception as e:
        print(f"AI分析失败: {e}，标题: {title[:30]}...")
        return default

# ==================== 行业/概念热度计算 ====================
def calculate_heat(news_df, half_life_days=HALF_LIFE_DAYS):
    """根据新闻计算行业和概念的热度（考虑情感和时间衰减）"""
    import ast
    def clean_list(col_val):
        if isinstance(col_val, str):
            try:
                lst = ast.literal_eval(col_val)
            except:
                lst = []
        elif isinstance(col_val, list):
            lst = col_val
        else:
            lst = []
        return [item for item in lst if isinstance(item, str) and len(item) >= 2]

    news_df['ai_industries'] = news_df['ai_industries'].apply(clean_list)
    news_df['ai_concepts'] = news_df['ai_concepts'].apply(clean_list)

    news_df['pub_datetime'] = news_df['time_raw'].apply(parse_news_time)
    news_df['weight'] = news_df['pub_datetime'].apply(lambda dt: time_decay_weight(dt, half_life_days))

    industry_scores = {}
    industry_pos = {}
    industry_neg = {}
    concept_scores = {}
    concept_pos = {}
    concept_neg = {}

    for _, row in news_df.iterrows():
        w = row['weight']
        sentiment = row.get('sentiment', 0.0)
        impact = row.get('impact', 0)
        effect = w * impact * abs(sentiment)
        industries = row.get('ai_industries', [])
        concepts = row.get('ai_concepts', [])
        if sentiment > 0:
            for ind in industries:
                industry_pos[ind] = industry_pos.get(ind, 0) + effect
                industry_scores[ind] = industry_scores.get(ind, 0) + effect
            for con in concepts:
                concept_pos[con] = concept_pos.get(con, 0) + effect
                concept_scores[con] = concept_scores.get(con, 0) + effect
        elif sentiment < 0:
            for ind in industries:
                industry_neg[ind] = industry_neg.get(ind, 0) + effect
                industry_scores[ind] = industry_scores.get(ind, 0) - effect
            for con in concepts:
                concept_neg[con] = concept_neg.get(con, 0) + effect
                concept_scores[con] = concept_scores.get(con, 0) - effect

    industry_df = pd.DataFrame([
        {'行业': k, '热度得分': v, '利好得分': industry_pos.get(k, 0), '利空得分': industry_neg.get(k, 0)}
        for k, v in industry_scores.items()
    ]).sort_values('热度得分', ascending=False).reset_index(drop=True)

    concept_df = pd.DataFrame([
        {'概念': k, '热度得分': v, '利好得分': concept_pos.get(k, 0), '利空得分': concept_neg.get(k, 0)}
        for k, v in concept_scores.items()
    ]).sort_values('热度得分', ascending=False).reset_index(drop=True)

    return industry_df, concept_df

# ==================== 股票数据获取（含缓存）====================
_DAILY_CACHE = {}
_FINA_INDICATOR_CACHE = {}

def get_last_trade_date():
    """获取最近一个交易日（简化版：判断周末，不考虑法定节假日）"""
    today = datetime.now()
    if today.weekday() == 5:
        last = today - timedelta(days=1)
    elif today.weekday() == 6:
        last = today - timedelta(days=2)
    else:
        last = today
    return last.strftime('%Y%m%d')

def load_daily_basic():
    """加载每日基本面数据，优先从本地缓存读取"""
    cache_file = 'daily_basic_cache.csv'
    last_trade = get_last_trade_date()
    if os.path.exists(cache_file):
        cached_df = pd.read_csv(cache_file)
        if 'trade_date' in cached_df.columns and cached_df['trade_date'].iloc[0] == last_trade:
            print(f"使用缓存数据，交易日: {last_trade}")
            return cached_df.drop(columns=['trade_date'])
    print(f"获取新的 daily_basic 数据，交易日: {last_trade}")
    try:
        df = pro.daily_basic(trade_date=last_trade, fields='ts_code,pe,pb,turnover_rate,volume_ratio')
        if df is not None and not df.empty:
            df['trade_date'] = last_trade
            df.to_csv(cache_file, index=False, encoding='utf-8-sig')
            return df.drop(columns=['trade_date'])
        else:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            df = pro.daily_basic(trade_date=yesterday, fields='ts_code,pe,pb,turnover_rate,volume_ratio')
            if df is not None and not df.empty:
                df['trade_date'] = yesterday
                df.to_csv(cache_file, index=False, encoding='utf-8-sig')
                return df.drop(columns=['trade_date'])
            else:
                print("无法获取 daily_basic 数据")
                return pd.DataFrame()
    except Exception as e:
        print(f"获取 daily_basic 失败: {e}")
        return pd.DataFrame()

def get_stock_fundamental(ts_code):
    if ts_code in _FINA_INDICATOR_CACHE:
        return _FINA_INDICATOR_CACHE[ts_code]
    try:
        df = pro.fina_indicator(ts_code=ts_code, fields='ts_code,roe,eps')
        if df is not None and not df.empty:
            latest = df.iloc[0]
            _FINA_INDICATOR_CACHE[ts_code] = latest
            return latest
        else:
            return None
    except Exception as e:
        print(f"获取财务指标失败 {ts_code}: {e}")
        return None

def get_stock_technical(ts_code, days=5):
    cache_key = f"{ts_code}_{days}"
    if cache_key in _DAILY_CACHE:
        return _DAILY_CACHE[cache_key]
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date,
                       fields='trade_date,close,pct_chg')
        if df is not None and not df.empty:
            df = df.sort_values('trade_date')
            if len(df) >= days:
                close_now = df['close'].iloc[-1]
                close_days_ago = df['close'].iloc[-days]
                ret_days = (close_now - close_days_ago) / close_days_ago * 100
            else:
                ret_days = None
            avg_pct = df['pct_chg'].tail(days).mean() if len(df) >= days else None
            result = {'return_days': ret_days, 'avg_pct_chg': avg_pct}
            _DAILY_CACHE[cache_key] = result
            return result
        else:
            return None
    except Exception as e:
        print(f"获取技术指标失败 {ts_code}: {e}")
        return None

# ==================== 股票综合排名（支持行业/概念筛选）====================
def rank_stocks(industry_heat, concept_heat, filter_industries=None, filter_concepts=None, top_n=10):
    """
    优化版：先用基础指标快速筛选前100只，再对候选股获取ROE及技术指标进行精细评分，并生成推荐理由
    新增参数 filter_concepts：概念关键词列表
    """
    stock_basic = get_stock_basic()
    daily_basic = load_daily_basic()
    if daily_basic.empty:
        print("无法获取 daily_basic 数据，返回空结果")
        return pd.DataFrame(columns=['股票代码', '股票名称', '综合得分', '推荐理由'])

    hot_industries = set(industry_heat.head(5)['行业'].tolist()) if not industry_heat.empty else set()
    hot_concepts = set(concept_heat.head(5)['概念'].tolist()) if not concept_heat.empty else set()

    # 第一轮：快速筛选（行业 + 概念 + 基本面）
    candidates = []
    for _, row in stock_basic.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        industry = row['industry']

        # 行业筛选（如果设置了）
        if filter_industries:
            ind_matched = False
            for f_ind in filter_industries:
                if f_ind.lower() in industry.lower():
                    ind_matched = True
                    break
            if not ind_matched:
                continue

        # 概念筛选（如果设置了）
        if filter_concepts:
            con_matched = False
            # 尝试精确映射
            if USE_PRECISE_CONCEPT:
                for f_con in filter_concepts:
                    for concept_name, stock_set in CONCEPT_STOCK_MAP.items():
                        if f_con.lower() in concept_name.lower() and ts_code in stock_set:
                            con_matched = True
                            break
                    if con_matched:
                        break
            # 如果精确映射未匹配，则尝试回退匹配（同时匹配股票名称和行业）
            if not con_matched:
                name_lower = name.lower()
                industry_lower = industry.lower()
                for f_con in filter_concepts:
                    kw = f_con.lower()
                    if kw in name_lower or kw in industry_lower:
                        con_matched = True
                        break
            if not con_matched:
                continue

        sd = daily_basic[daily_basic['ts_code'] == ts_code]
        if sd.empty:
            continue

        pe = sd['pe'].values[0] if 'pe' in sd.columns else np.nan
        pb = sd['pb'].values[0] if 'pb' in sd.columns else np.nan
        turnover = sd['turnover_rate'].values[0] if 'turnover_rate' in sd.columns else np.nan

        quick_score = 0.0
        if pd.notna(pe) and pe > 0:
            quick_score += max(0, 50 - pe) / 50 * 20
        if pd.notna(pb) and pb > 0:
            quick_score += max(0, 10 - pb) / 10 * 10
        if pd.notna(turnover):
            if turnover < 1:
                quick_score += 0
            elif turnover > 10:
                quick_score += 5
            else:
                quick_score += turnover * 0.5

        # 行业热度加分（如果股票行业与热门行业匹配）
        hot_match = any(hot_ind in industry for hot_ind in hot_industries)
        if hot_match and not industry_heat.empty:
            hot_score_mean = industry_heat[industry_heat['行业'].isin(hot_industries)]['热度得分'].mean()
            if pd.notna(hot_score_mean):
                quick_score += hot_score_mean * 0.1

        candidates.append({
            'ts_code': ts_code,
            'name': name,
            'quick_score': quick_score,
            'pe': pe,
            'pb': pb,
            'turnover': turnover,
            'industry': industry,
            'hot_match': hot_match
        })

    if not candidates:
        print("没有找到符合条件的候选股票。")
        return pd.DataFrame(columns=['股票代码', '股票名称', '综合得分', '推荐理由'])

    # 按快速得分排序，取前100名进入第二轮
    candidates.sort(key=lambda x: x['quick_score'], reverse=True)
    top_candidates = candidates[:100]

    # 第二轮：精细评分（加入ROE和技术指标）
    final_results = []
    for item in top_candidates:
        ts_code = item['ts_code']
        name = item['name']
        pe = item['pe']
        pb = item['pb']
        turnover = item['turnover']
        industry = item['industry']
        hot_match = item['hot_match']

        fina = get_stock_fundamental(ts_code)
        roe = fina['roe'] if fina is not None and 'roe' in fina else np.nan
        tech = get_stock_technical(ts_code, days=5)
        ret_5d = tech['return_days'] if tech and tech['return_days'] is not None else np.nan

        final_score = 0.0
        if pd.notna(pe) and pe > 0:
            final_score += max(0, 50 - pe) / 50 * 20
        if pd.notna(pb) and pb > 0:
            final_score += max(0, 10 - pb) / 10 * 10
        if pd.notna(roe):
            final_score += min(roe, 30) / 30 * 15
        if pd.notna(turnover):
            if turnover < 1:
                final_score += 0
            elif turnover > 10:
                final_score += 5
            else:
                final_score += turnover * 0.5
        if pd.notna(ret_5d):
            if ret_5d > 20:
                final_score += 10
            elif ret_5d < -10:
                final_score -= 5
            else:
                final_score += ret_5d * 0.5

        # 生成推荐理由
        reason_parts = []
        if filter_industries and any(fi.lower() in industry.lower() for fi in filter_industries):
            reason_parts.append("您关注的行业")
        if filter_concepts:
            # 检查股票是否属于任何关注的概念（用于理由展示）
            if USE_PRECISE_CONCEPT:
                matched_concepts = []
                for fc in filter_concepts:
                    for concept_name, stock_set in CONCEPT_STOCK_MAP.items():
                        if fc.lower() in concept_name.lower() and ts_code in stock_set:
                            matched_concepts.append(concept_name)
                            break
                if matched_concepts:
                    reason_parts.append(f"概念相关({', '.join(matched_concepts)})")
            else:
                matched_concepts = []
                name_lower = name.lower()
                for fc in filter_concepts:
                    if fc.lower() in name_lower:
                        matched_concepts.append(fc)
                if matched_concepts:
                    reason_parts.append(f"概念相关({', '.join(matched_concepts)})")
        if hot_match:
            reason_parts.append("行业热点")
        if pd.notna(roe) and roe > 15:
            reason_parts.append("ROE高")
        if pd.notna(pe) and pe < 15:
            reason_parts.append("低估值")
        if pd.notna(turnover) and turnover > 5:
            reason_parts.append("交易活跃")
        if pd.notna(ret_5d) and ret_5d > 10:
            reason_parts.append("近期强势")
        if not reason_parts:
            reason = "基本面稳健"
        else:
            reason = "，".join(reason_parts)

        final_results.append({
            '股票代码': ts_code,
            '股票名称': name,
            '综合得分': round(final_score, 2),
            '推荐理由': reason
        })

    if not final_results:
        print("所有候选股票在精细评分后均被排除，无结果返回。")
        return pd.DataFrame(columns=['股票代码', '股票名称', '综合得分', '推荐理由'])

    result_df = pd.DataFrame(final_results).sort_values('综合得分', ascending=False).head(top_n)
    return result_df

# ==================== 新增交互功能函数 ====================
NEWS_DATA_FILE = 'news_data.csv'
filter_industries = None  # 全局行业筛选条件
filter_concepts = None    # 全局概念筛选条件

def run_analysis_flow(force_refresh=False):
    """执行完整分析流程：增量爬取新闻、AI分析、热度计算、推荐生成，并自动限制新闻总数不超过1000条"""
    global crawled_titles, news_df, industry_heat, concept_heat, top_stocks, filter_industries, filter_concepts

    if force_refresh:
        print("强制刷新模式：清空已爬取标题和历史数据...")
        if os.path.exists(CRAWLED_TITLES_FILE):
            os.remove(CRAWLED_TITLES_FILE)
        if os.path.exists(NEWS_DATA_FILE):
            os.remove(NEWS_DATA_FILE)
        crawled_titles = set()
    else:
        crawled_titles = load_crawled_titles()

    if os.path.exists(NEWS_DATA_FILE):
        history_df = pd.read_csv(NEWS_DATA_FILE)
        print(f"加载历史新闻数据 {len(history_df)} 条")
    else:
        history_df = pd.DataFrame()
        print("无历史新闻数据，将强制爬取初始数据...")
        crawled_titles = set()

    print("\n[步骤1] 增量爬取财经新闻...")
    df_east_new = crawl_eastmoney_with_playwright(page_num_limit=5, existing_titles=crawled_titles)
    df_ak_new = get_akshare_news_deduplicated(days=2, existing_titles=crawled_titles)

    new_news_list = [df for df in [df_east_new, df_ak_new] if not df.empty]
    if new_news_list:
        new_news_df = pd.concat(new_news_list, ignore_index=True)
        new_news_df = new_news_df.drop_duplicates(subset=['title'])
        print(f"本次新增新闻共 {len(new_news_df)} 条")

        print("\n[步骤2] 抓取新新闻正文...")
        new_news_df['content'] = new_news_df.apply(
            lambda row: fetch_news_content(row['link']) if row['source'] == 'eastmoney' and not row.get('content') else row.get('content', ''),
            axis=1
        )

        print("\n[步骤3] AI智能体分析新新闻...")
        ai_results = []
        for idx, row in new_news_df.iterrows():
            print(f"正在分析第 {idx+1}/{len(new_news_df)} 条: {row['title'][:30]}...")
            result = analyze_news_with_ai(row['title'], row['content'])
            ai_results.append(result)
            time.sleep(0.5)

        ai_df = pd.DataFrame(ai_results)
        for col in ['sentiment', 'industries', 'concepts', 'impact', 'companies', 'news_type']:
            if col not in ai_df.columns:
                if col in ['sentiment', 'impact']:
                    ai_df[col] = 0.0
                else:
                    ai_df[col] = [[] for _ in range(len(ai_df))]
        ai_df.rename(columns={
            'industries': 'ai_industries',
            'concepts': 'ai_concepts',
            'companies': 'ai_companies'
        }, inplace=True)
        new_news_df = pd.concat([new_news_df, ai_df], axis=1)

        if not history_df.empty:
            combined_df = pd.concat([history_df, new_news_df], ignore_index=True)
        else:
            combined_df = new_news_df

        # 去重
        combined_df = combined_df.drop_duplicates(subset=['title'])

        # ===== 新增：限制总条数最多1000条 =====
        if len(combined_df) > 1000:
            # 解析时间以便排序
            combined_df['sort_time'] = combined_df['time_raw'].apply(parse_news_time)
            # 按时间降序排序，最新的在前
            combined_df = combined_df.sort_values('sort_time', ascending=False, na_position='last')
            combined_df = combined_df.head(1000)
            combined_df = combined_df.drop(columns=['sort_time'])
            print(f"截断后保留最新的1000条")
        # 更新已爬取标题文件：用保留的所有新闻标题覆盖
        all_retained_titles = {str(t) for t in combined_df['title'].tolist() if pd.notna(t)}
        save_crawled_titles(all_retained_titles, append=False)

        # 保存合并后的数据到缓存
        combined_df.to_csv(NEWS_DATA_FILE, index=False, encoding='utf-8-sig')
        print(f"新闻数据已保存至 {NEWS_DATA_FILE}，总条数 {len(combined_df)}")
        news_df = combined_df
    else:
        if history_df.empty:
            print("没有新新闻，且无历史数据，无法分析。")
            return None, None, None, None
        else:
            print("没有新新闻，将使用历史新闻数据进行分析。")
            news_df = history_df
            # 无新新闻时，不修改标题文件（历史数据未变）

    print("\n[步骤4] 计算行业与概念热度...")
    industry_heat, concept_heat = calculate_heat(news_df)
    if not industry_heat.empty:
        print("行业热度排名（前10）：")
        print(industry_heat.head(10).to_string(index=False))
    if not concept_heat.empty:
        print("\n概念热度排名（前10）：")
        print(concept_heat.head(10).to_string(index=False))

    print("\n[步骤5] 股票综合排名...")
    top_stocks = rank_stocks(industry_heat, concept_heat, filter_industries=filter_industries, filter_concepts=filter_concepts, top_n=10)
    if not top_stocks.empty:
        print("\n【推荐股票】Top 10")
        print(top_stocks.to_string(index=False))
        top_stocks.to_csv('推荐股票.csv', index=False, encoding='utf-8-sig')
    else:
        print("未选出符合条件的股票。")

    # 导出热度数据供网站使用（无论是否有推荐股票，只要热度存在就导出）
    if not industry_heat.empty:
        industry_heat.to_csv('industry_heat.csv', index=False, encoding='utf-8-sig')
    if not concept_heat.empty:
        concept_heat.to_csv('concept_heat.csv', index=False, encoding='utf-8-sig')

    return industry_heat, concept_heat, news_df, top_stocks

def search_stock(keyword):
    """根据名称或代码模糊搜索股票，并可选择预测，同时绘制K线图"""
    stock_basic = get_stock_basic()
    mask = (stock_basic['name'].str.contains(keyword, na=False) |
            stock_basic['symbol'].str.contains(keyword, na=False) |
            stock_basic['ts_code'].str.contains(keyword, na=False))
    result = stock_basic[mask]
    if result.empty:
        print("未找到匹配的股票。")
        return result

    print(f"找到 {len(result)} 只股票：")
    print(result[['ts_code', 'name', 'industry']].to_string(index=False))

    # 如果只找到一只，直接询问是否预测；如果多只，让用户选择
    if len(result) == 1:
        ts_code = result.iloc[0]['ts_code']
        pred_choice = input(f"\n是否对 {ts_code} 进行股价预测并查看K线图？(y/n，默认n): ").strip().lower()
        if pred_choice == 'y':
            news_df = load_news_data()
            pred_result = predict_stock(ts_code, news_df)
            print("\n【预测结果】")
            for k, v in pred_result.items():
                print(f"{k}: {v}")
            # 绘制K线图
            df = get_daily_data(ts_code)
            if not df.empty and len(df) >= 20:
                name = result.iloc[0]['name']
                plot_kline(ts_code, df, name)
    else:
        # 多只时，询问用户输入要预测的股票代码
        code_input = input("\n请输入您想要预测的股票代码（直接回车跳过预测）：").strip()
        if code_input:
            # 检查输入是否在结果中
            matched = result[result['ts_code'].str.contains(code_input) | result['symbol'].str.contains(code_input)]
            if not matched.empty:
                ts_code = matched.iloc[0]['ts_code']
                name = matched.iloc[0]['name']
                news_df = load_news_data()
                pred_result = predict_stock(ts_code, news_df)
                print("\n【预测结果】")
                for k, v in pred_result.items():
                    print(f"{k}: {v}")
                # 绘制K线图
                df = get_daily_data(ts_code)
                if not df.empty and len(df) >= 20:
                    plot_kline(ts_code, df, name)
            else:
                print("输入的股票代码不在搜索结果中，跳过预测。")
    return result

def update_filters():
    """交互式修改行业和概念筛选关键词"""
    global filter_industries, filter_concepts
    print("当前行业筛选关键词：", filter_industries if filter_industries else "无")
    print("当前概念筛选关键词：", filter_concepts if filter_concepts else "无")
    print("\n请输入行业关键词（多个可用空格或逗号分隔，直接回车保持原样）：")
    ind_input = input().strip()
    if ind_input:
        import re
        keywords = re.split(r'[ ,，]+', ind_input)
        filter_industries = [k.strip() for k in keywords if k.strip()]
    # 如果直接回车，保留原值（不变）

    print("\n请输入概念关键词（多个可用空格或逗号分隔，直接回车保持原样）：")
    con_input = input().strip()
    if con_input:
        import re
        keywords = re.split(r'[ ,，]+', con_input)
        filter_concepts = [k.strip() for k in keywords if k.strip()]
    # 如果直接回车，保留原值

    print(f"已更新行业关键词：{filter_industries}")
    print(f"已更新概念关键词：{filter_concepts}")

import sys

# 在云端也尝试加载概念映射缓存（如果存在）
if os.path.exists(CONCEPT_CACHE_FILE):
    try:
        with open(CONCEPT_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        CONCEPT_STOCK_MAP = {k: set(v) for k, v in cache_data.items()}
        USE_PRECISE_CONCEPT = True
        print("已从缓存文件加载概念映射")
    except Exception as e:
        print(f"读取概念缓存失败: {e}")

if len(sys.argv) > 1 and sys.argv[1] == '--auto':
    # 自动运行模式
    print("自动运行模式：执行一次完整分析...")
    result = run_analysis_flow(force_refresh=False)
    print("分析完成。")
    sys.exit(0)
if not (os.getenv('STREAMLIT_RUNTIME') or os.getenv('GITHUB_ACTIONS')):
    load_concept_map_from_akshare()

# ==================== 主程序（交互菜单）====================
if __name__ == "__main__":
    print("=" * 50)
    print("智能体选股原型 v3.5 交互式菜单版（集成预测，使用AkShare概念映射 + 缓存）")
    print("=" * 50)

    # 首次运行自动执行完整分析
    print("\n首次运行，执行完整分析流程...")
    industry_heat = concept_heat = news_df = top_stocks = None
    result = run_analysis_flow(force_refresh=False)
    if result[0] is not None:
        industry_heat, concept_heat, news_df, top_stocks = result

    # 交互菜单
    while True:
        print("\n" + "=" * 40)
        print("请选择操作：")
        print("1. 更新数据（重新爬取新闻并分析）")
        print("2. 搜索股票（并可选预测）")
        print("3. 调整行业/概念偏好")
        print("4. 重新生成推荐（基于当前数据和偏好）")
        print("5. 预测当前推荐股票")
        print("6. 生成推荐并自动预测")
        print("0. 退出程序")
        choice = input("请输入数字选择：").strip()

        if choice == '1':
            print("\n开始更新数据...")
            force = input("是否强制刷新全部历史数据？(y/n，默认n): ").strip().lower() == 'y'
            result = run_analysis_flow(force_refresh=force)
            if result[0] is not None:
                industry_heat, concept_heat, news_df, top_stocks = result

        elif choice == '2':
            keyword = input("请输入股票名称或代码关键字：").strip()
            if keyword:
                search_stock(keyword)
            else:
                print("输入无效。")

        elif choice == '3':
            update_filters()
            print("偏好已更新，可使用选项4重新生成推荐。")

        elif choice == '4':
            if news_df is None or industry_heat is None:
                print("数据未加载，请先执行更新数据（选项1）。")
            else:
                print("\n重新生成推荐...")
                top_stocks = rank_stocks(industry_heat, concept_heat, filter_industries=filter_industries, filter_concepts=filter_concepts, top_n=10)
                if not top_stocks.empty:
                    print("\n【推荐股票】Top 10")
                    print(top_stocks.to_string(index=False))
                    top_stocks.to_csv('推荐股票.csv', index=False, encoding='utf-8-sig')
                else:
                    print("未选出符合条件的股票。")

        elif choice == '5':
            if top_stocks is not None and not top_stocks.empty:
                print("\n开始对推荐股票进行预测...")
                stock_list = top_stocks['股票代码'].tolist()
                from stock_predictor import run_prediction
                run_prediction(stock_list=stock_list)
            else:
                print("当前没有推荐股票，请先执行选股（选项1或4）。")

        elif choice == '6':
            if news_df is None or industry_heat is None:
                print("数据未加载，请先执行更新数据（选项1）。")
            else:
                print("\n重新生成推荐...")
                top_stocks = rank_stocks(industry_heat, concept_heat, filter_industries=filter_industries, filter_concepts=filter_concepts, top_n=10)
                if not top_stocks.empty:
                    print("\n【推荐股票】Top 10")
                    print(top_stocks.to_string(index=False))
                    top_stocks.to_csv('推荐股票.csv', index=False, encoding='utf-8-sig')
                    print("\n开始对推荐股票进行预测...")
                    stock_list = top_stocks['股票代码'].tolist()
                    from stock_predictor import run_prediction
                    run_prediction(stock_list=stock_list)
                else:
                    print("未选出符合条件的股票。")

        elif choice == '0':
            print("感谢使用，再见！")
            break

        else:
            print("无效输入，请重新选择。")
