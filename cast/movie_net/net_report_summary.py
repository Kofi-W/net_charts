# streamlit 报告页面
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# 原始数据读取
df = pd.read_csv("../data/movie_data_rated.csv")
df['k_cast_id'] = df['k_cast_id'].apply(lambda x: str(x))


st.title("电影-影人 网络分析报告")

# 输入影人姓名查询


desc = """国内类似优映文化的创作团队及郭靖宇杨志刚模式解析
要判断国内哪些团队类似优映文化，首先需要明确优映模式的核心特征：以固定核心主创（编导 + 演员）为基石，坚持原创剧本与密集反转的叙事风格，深耕特定题材构建 IP 宇宙，以小成本撬动高口碑，拒绝流量明星、专注内容本身，形成独特的创作生态与观众文化。
一、国内类似优映模式的核心团队
1. 郭靖宇 “郭家班”（长信传媒）
以郭靖宇为核心，联合妻子岳丽娜（出品人 + 演员）、弟弟杨志刚（核心主演）、导演柏杉 / 巨兴茂、编剧魏风华 / 小吉祥天等形成稳定班底。从《铁梨花》《打狗棍》到《灵魂摆渡》《唐朝诡事录》，坚持原创 IP 开发，构建了 “中式悬疑探案宇宙” 与 “年代传奇宇宙”。团队特点是家族式紧密协作，不依赖流量明星，经费优先投入制作，形成了 “非流量、强表演、重叙事” 的鲜明风格，与优映在 “内容为王” 理念上高度契合。
2. 五元文化与弧光联盟
由导演五百创立，以《白夜追凶》《隐秘而伟大》等悬疑作品树立行业标杆。五元文化坚持固定导演团队 + 类型深耕，发起的 “弧光联盟” 汇聚了王伟、杨苗等多位风格统一的悬疑剧导演，形成 “联盟创作 + IP 共享” 的生态模式。与优映相似，五元文化注重剧本打磨与逻辑严谨，擅长以强情节、多反转吸引观众，同样拒绝流量至上，偏好实力派演员。
3. 正午阳光（孔笙 / 侯鸿亮 / 李雪铁三角）
脱胎于山影，以 “国剧门脸” 著称。核心班底稳定，孔笙、侯鸿亮、李雪构成创作核心，演员方面形成 “正午熟脸” 矩阵（如刘钧、刘琳、王凯等）。团队坚持现实主义创作 + 品质至上，虽题材多元（从《琅琊榜》到《山海情》），但始终以扎实剧本与精湛制作取胜，与优映共享 “内容为王” 的创作初心，区别在于正午阳光更偏向工业化大制作，而优映保留草根创作的灵活性。
4. 灵魂摆渡团队（小吉祥天 + 巨兴茂）
作为 “郭家班” 的重要分支，由编剧小吉祥天与导演巨兴茂组成核心创作组，打造了《灵魂摆渡》系列这一 “国产灵异天花板”。团队以原创世界观 + 单元剧叙事构建灵异宇宙，擅长 “鬼事讲人心” 的隐喻手法，演员于毅、刘智扬、肖茵长期固定，形成了与优映相似的 “铁三角” 创作模式，同样以小成本实现口碑逆袭。
5. 艺画开天（动画领域类似团队）
在国漫领域，艺画开天以《疯味英雄》《灵笼》等作品构建原创科幻宇宙，核心团队长期稳定，坚持 “硬核科幻 + 深度世界观” 的创作理念。与优映相似，艺画开天从草根团队起步，以高质量内容打破行业偏见，形成独特的粉丝文化，区别在于其专注动画赛道，注重技术与艺术的结合。
二、郭靖宇杨志刚模式与优映模式的异同
1. 高度契合的核心特质
固定班底 + 紧密协作：两者均以核心主创为中心形成稳定团队，优映有李洪绸 / 邢冬冬 / 车志刚 + 邵庄 / 安宁 / 杨羽，郭家班有郭靖宇 + 杨志刚 / 岳丽娜 + 柏杉 / 巨兴茂，长期合作保障了风格一致性与创作效率。
原创至上 + 内容为王：都拒绝 IP 改编依赖，坚持剧本原创，郭靖宇把控《唐诡》系列创作方向，优映 “出奇创作组” 集体打磨每一句台词，均以叙事质量为第一标准。
反流量思维：均不迷信流量明星，优映用素人演员，郭家班用实力派与熟人班底，将资金集中于制作环节，形成 “小成本高口碑” 的创作路径。
IP 宇宙构建：优映打造 “泛毛骗宇宙”，郭家班构建 “唐诡宇宙”“灵魂摆渡宇宙”，通过跨作品联动与伏笔回收增强 IP 粘性。
2. 显著差异的创作生态
家族化 vs 学院派：郭家班呈现家族式创作特征（兄弟、夫妻共同参与），情感纽带更强；优映源自河北影视传媒学院学生团队，是学院派草根创业的典型，团队关系更偏向同窗协作。
题材侧重不同：优映专注现代悬疑 + 街头智斗，擅长用现实逻辑构建反转；郭家班则覆盖年代传奇 + 古装探案 + 灵异奇幻，题材更广泛，尤其擅长将悬疑与传统文化融合。
制作规模差异：优映从 6 万元小成本起步，逐步升级制作但仍保持轻量化；郭家班则实现了工业化制作升级，《唐诡》系列在服化道、场景搭建上投入更大，制作更精细。
演员定位不同：杨志刚作为郭靖宇弟弟，是郭家班核心男主，角色量身定制；优映演员则无绝对主角，更偏向群像叙事，角色与演员相互成就而非单方面塑造。
3. 结论：郭靖宇杨志刚算类似优映的模式
郭靖宇杨志刚团队（郭家班）与优映文化在核心创作理念、团队运作模式、反流量策略与 IP 宇宙构建等关键维度高度相似，均属于国内影视行业中 “以内容为核心、以固定班底为支撑” 的优质创作团队。两者的差异更多体现在题材选择、制作规模与团队构成方式上，而非本质创作逻辑的不同。
三、其他团队的差异化特征
正午阳光：虽固定班底与内容至上理念与优映一致，但更偏向主流价值观表达与工业化大制作，题材多元且更具普适性，与优映的小众悬疑风格形成区别。
五元文化：以导演联盟形式扩大创作版图，悬疑风格更偏向现代都市与犯罪类型，与优映的街头骗术 + 奇幻元素的复合风格不同。
艺画开天：专注动画赛道，通过科幻世界观构建与技术升级形成差异化竞争力，与优映的真人影视创作路径有明显区别。
综上，国内类似优映文化的团队以郭靖宇郭家班最为典型，此外五元文化、正午阳光、灵魂摆渡团队等也在不同维度呈现出相似的创作特征，共同构成了国产影视中 “内容为王” 的创作力量。
"""


# 页面配置
st.set_page_config(
    page_title="影视团队模式解析：优映文化 VS 郭家班",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义 CSS 增强卡片视觉效果
st.markdown("""
    <style>
    .big-font { font-size: 20px !important; font-weight: bold; }
    .sub-font { font-size: 14px; color: #555; }
    .card-header { font-weight: bold; font-size: 18px; margin-bottom: 10px; display: block;}
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 5px; color: white;}
    .tag-blue { background-color: #3b82f6; }
    .tag-green { background-color: #10b981; }
    .tag-red { background-color: #ef4444; }
    .tag-purple { background-color: #8b5cf6; }
    </style>
""", unsafe_allow_html=True)

# 标题区
st.title("🎬 国内影视创作团队模式解析")
st.markdown("### —— 基于“优映文化”视角的深度横向测评")
st.divider()

# --- 第一部分：基石定义 ---
st.subheader("🎯 标尺：优映模式核心特征")
with st.container(border=True):
    cols = st.columns(4)
    features = [
        ("核心基石", "固定编导 + 演员班底", "👥"),
        ("叙事风格", "原创剧本 + 密集反转", "📝"),
        ("运营策略", "小成本撬动高口碑 + 拒绝流量", "💰"),
        ("长期主义", "深耕题材 + 构建 IP 宇宙", "🌍")
    ]
    for col, (title, desc, icon) in zip(cols, features):
        with col:
            st.markdown(f"### {icon} {title}")
            st.caption(desc)

st.divider()

# --- 第二部分：核心对决 (优映 vs 郭家班) ---
st.subheader("⚔️ 核心对比：郭靖宇/杨志刚(郭家班) VS 优映文化")

# 结论先行
st.info("💡 **结论**：郭家班是国内与优映文化在“核心创作理念”、“团队运作模式”上高度相似的团队，差异仅在于题材与工业化规模。")

col1, col2 = st.columns(2)

# 左侧：高度契合点
with col1:
    with st.container(border=True):
        st.markdown("#### 🤝 高度契合的核心特质")
        
        st.markdown("**1. 固定班底 + 紧密协作**")
        st.markdown("""
        * **优映**: 李洪绸 / 邢冬冬 + 邵庄 / 安宁
        * **郭家班**: 郭靖宇 / 岳丽娜 + 杨志刚 / 巨兴茂
        > 长期合作保障风格一致性与效率。
        """)
        
        st.markdown("**2. 原创至上 + 内容为王**")
        st.markdown("""
        * 均拒绝依赖 IP 改编，坚持原创剧本。
        * 《唐诡》与《毛骗》均以叙事质量为第一标准。
        """)
        
        st.markdown("**3. 反流量思维**")
        st.markdown("""
        * 资金集中于制作，不迷信流量明星。
        * **路径**: 素人/熟人班底 → 小成本 → 高口碑。
        """)
        
        st.markdown("**4. IP 宇宙构建**")
        st.markdown("""
        * **优映**: 泛毛骗宇宙
        * **郭家班**: 唐诡宇宙 / 灵魂摆渡宇宙
        """)

# 右侧：差异化生态
with col2:
    with st.container(border=True):
        st.markdown("#### ⚖️ 显著差异的创作生态")
        
        # 使用表格或分列展示对比
        diff_data = {
            "维度": ["团队纽带", "题材侧重", "制作规模", "演员定位"],
            "郭家班 (长信)": ["家族式 (血缘/姻亲)", "年代传奇/古装探案/灵异", "工业化升级 (大投入)", "核心大男主 (量身定制)"],
            "优映文化": ["学院派 (同窗校友)", "现代悬疑 + 街头智斗", "作坊式 -> 轻量化", "群像叙事 (相互成就)"]
        }
        
        # 模拟卡片内的对比条目
        for i in range(4):
            dim = diff_data["维度"][i]
            guo = diff_data["郭家班 (长信)"][i]
            you = diff_data["优映文化"][i]
            
            st.write(f"**{dim}**")
            c_a, c_b = st.columns([1, 1])
            c_a.info(f"🦁 **郭**: {guo}")
            c_b.success(f"🦊 **优**: {you}")

st.divider()

# --- 第三部分：行业版图 ---
st.subheader("🗺️ 行业版图：其他相似团队解析")

team_cols = st.columns(3)

# 团队 1: 五元文化
with team_cols[0]:
    with st.container(border=True):
        st.markdown('<span class="card-header">🏙️ 五元文化 & 弧光联盟</span>', unsafe_allow_html=True)
        st.markdown('<span class="tag tag-blue">悬疑深耕</span><span class="tag tag-blue">导演联盟</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**核心人物**: 五百 (王伟/杨苗)")
        st.markdown("**代表作**: 《白夜追凶》《隐秘而伟大》")
        st.markdown("**相似点**: 拒绝流量，强情节多反转，剧本逻辑严谨。")
        st.warning("**差异**: 导演联盟模式 (非单一核心)，风格偏现代都市犯罪。")

# 团队 2: 正午阳光
with team_cols[1]:
    with st.container(border=True):
        st.markdown('<span class="card-header">☀️ 正午阳光</span>', unsafe_allow_html=True)
        st.markdown('<span class="tag tag-red">国剧门脸</span><span class="tag tag-red">正午熟脸</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**核心人物**: 孔笙 / 侯鸿亮 / 李雪")
        st.markdown("**代表作**: 《琅琊榜》《山海情》")
        st.markdown("**相似点**: 铁三角班底，内容为王，品质至上。")
        st.warning("**差异**: 工业化大制作，主流价值观表达，题材普适性强。")

# 团队 3: 灵魂摆渡 / 艺画开天
with team_cols[2]:
    with st.container(border=True):
        st.markdown('<span class="card-header">👻 灵魂摆渡 & 🚀 艺画开天</span>', unsafe_allow_html=True)
        st.markdown('<span class="tag tag-purple">垂直赛道</span><span class="tag tag-purple">高口碑</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**灵摆团队**: 郭家班分支，专注灵异单元剧，以小博大典范。")
        st.markdown("**艺画开天**: 动画界的“优映”，草根起步，硬核科幻宇宙。")
        st.warning("**差异**: 一个是题材垂直分支，一个是动画赛道技术流。")

# 底部总结
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <strong>综上所述</strong>：国内类似优映文化的团队以郭靖宇郭家班最为典型，此外五元文化、正午阳光等也在不同维度呈现出“内容为王”的共性。
</div>
""", unsafe_allow_html=True)