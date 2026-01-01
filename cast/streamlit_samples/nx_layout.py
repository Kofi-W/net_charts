import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from typing import Callable, Tuple, Optional

# -----------------------------------------------------------------------------
# 1. 页面配置与 CSS 样式
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NetworkX Layout Gallery", layout="wide")

# 提取 CSS 到独立常量
CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .layout-card {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        padding: 20px;
        margin-bottom: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: auto; 
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .layout-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        border-color: #ff4b4b;
    }

    /* 标题部分 */
    .card-header {
        margin-bottom: 10px;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 10px;
        flex: 0 0 auto;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f1f1f;
        margin: 0;
    }

    /* 图片容器 */
    .img-container {
        flex: 0 0 auto;
        width: 100%;
        height: 250px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .img-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }

    /* 滚动区域 */
    .scroll-content {
        flex: 1 1 auto;
        overflow-y: auto;
        padding-right: 5px;
    }
    
    /* 滚动条美化 */
    .scroll-content::-webkit-scrollbar {
        width: 6px;
    }
    .scroll-content::-webkit-scrollbar-thumb {
        background-color: #ccc;
        border-radius: 4px;
    }

    /* 描述文字 */
    .desc-text {
        font-size: 0.95rem;
        color: #444;
        line-height: 1.5;
        margin-bottom: 15px;
    }
    
    .desc-text ul {
        padding-left: 20px;
        margin: 5px 0;
    }
    
    .desc-text p {
        margin-bottom: 8px;
    }

    /* 代码块区域 */
    .code-block {
        background-color: #f0f2f6;
        border-radius: 6px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #d63384;
        white-space: pre-wrap;
        word-break: break-all;
        border: 1px solid #dee2e6;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. 辅助工具：Matplotlib 转 Base64
# -----------------------------------------------------------------------------
def plot_to_base64(G, layout_func, **kwargs):
    # 关闭之前的绘图以防内存泄漏
    plt.close('all')
    fig, ax = plt.subplots(figsize=(4, 4))

    try:
        if layout_func == nx.shell_layout:
            nlist = [list(range(0, 5)), list(range(5, len(G)))]
            pos = layout_func(G, nlist=nlist)
        elif layout_func == nx.multipartite_layout:
            for i, node in enumerate(G.nodes()):
                G.nodes[node]['subset'] = i % 3
            pos = layout_func(G, subset_key='subset')
        else:
            try:
                pos = layout_func(G, seed=42, **kwargs)
            except TypeError:
                pos = layout_func(G, **kwargs)

        nx.draw(G,
                pos,
                with_labels=True,
                node_color='#FF4B4B',
                node_size=300,
                edge_color='#888888',
                font_color='white',
                font_size=10,
                ax=ax)

        ax.axis('off')
        fig.patch.set_alpha(0.0)

        buf = io.BytesIO()
        plt.savefig(buf,
                    format='png',
                    bbox_inches='tight',
                    pad_inches=0.1,
                    transparent=True)
        buf.seek(0)
        b64_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{b64_str}", None

    except Exception as e:
        plt.close(fig)
        return None, str(e)


# -----------------------------------------------------------------------------
# 3. 数据内容 (已清理缩进，确保HTML解析正确)
# -----------------------------------------------------------------------------
# 注意：这里的 desc 内部不要使用缩进，或者使用 textwrap.dedent 处理
layout_data = [{
    "name":
    "Spring Layout (弹簧布局)",
    "func":
    nx.spring_layout,
    "desc":
    """<p>这是 NetworkX 的<b>默认布局</b> (Fruchterman-Reingold 算法)。它模拟物理系统：节点被视为相互排斥的带电粒子，而边则像弹簧一样牵引连接的节点。</p><ul><li><b>适用场景：</b>想要自然地展示图的“聚类”结构，适合大多数中小规模的网络。</li><li><b>特点：</b>节点分布比较均匀，能很好地体现社区结构。</li></ul>""",
    "code":
    "G = nx.barbell_graph(m1=5, m2=3)\npos = nx.spring_layout(G, seed=42)\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}, {
    "name":
    "Circular Layout (圆形布局)",
    "func":
    nx.circular_layout,
    "desc":
    """<p>将所有节点排列在一个圆周上。</p><ul><li><b>适用场景：</b>查看连接密度，或者当节点地位平等时（如 P2P 网络）。</li><li><b>特点：</b>最容易看清边的分布情况（是否密集），但无法体现图的聚类结构。</li></ul>""",
    "code":
    "G = nx.barbell_graph(m1=5, m2=3)\npos = nx.circular_layout(G)\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}, {
    "name":
    "Kamada-Kawai Layout",
    "func":
    nx.kamada_kawai_layout,
    "desc":
    """<p>基于“图论距离”（最短路径距离）的力导向布局。试图使节点在绘图中的几何距离与它们在图中的理论距离成正比。</p><ul><li><b>适用场景：</b>需要高质量的全局结构展示。</li><li><b>特点：</b>通常比 Spring 更美观，区分不同分支，但计算成本较高。</li></ul>""",
    "code":
    "G = nx.barbell_graph(m1=5, m2=3)\npos = nx.kamada_kawai_layout(G)\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}, {
    "name":
    "Shell Layout (壳层布局)",
    "func":
    nx.shell_layout,
    "desc":
    """<p>将节点放置在同心圆（壳）上。需要指定哪些节点在哪个壳层。</p><ul><li><b>适用场景：</b>当图具有层级结构，或者想突出显示核心节点时。</li><li><b>特点：</b>需要手动指定 <code>nlist</code> 定义壳层。</li></ul>""",
    "code":
    "G = nx.barbell_graph(m1=5, m2=3)\nshells = [list(range(5)), list(range(5, len(G)))]\npos = nx.shell_layout(G, nlist=shells)\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}, {
    "name":
    "Random Layout (随机布局)",
    "func":
    nx.random_layout,
    "desc":
    """<p>将节点随机放置在单位正方形内。</p><ul><li><b>适用场景：</b>几乎不用于最终展示。通常用于算法测试或作为力导向算法的“初始状态”。</li><li><b>特点：</b>杂乱无章，没有结构信息。</li></ul>""",
    "code":
    "G = nx.barbell_graph(m1=5, m2=3)\npos = nx.random_layout(G, seed=10)\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}, {
    "name":
    "Spectral Layout (谱布局)",
    "func":
    nx.spectral_layout,
    "desc":
    """<p>利用图拉普拉斯矩阵的特征向量来确定节点坐标。</p><ul><li><b>适用场景：</b>识别图的对称性、特殊的几何结构或简单的聚类。</li><li><b>特点：</b>计算速度快（基于线性代数），但有时会导致节点重叠成一条线。</li></ul>""",
    "code":
    "G = nx.barbell_graph(m1=5, m2=3)\npos = nx.spectral_layout(G)\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}, {
    "name":
    "Spiral Layout (螺旋布局)",
    "func":
    nx.spiral_layout,
    "desc":
    """<p>将节点按照某种线性顺序排列成螺旋状。</p><ul><li><b>适用场景：</b>节点具有时间序列属性，或者具有线性的 ID 增长关系。</li><li><b>特点：</b>视觉效果独特，能展示长链条结构。</li></ul>""",
    "code":
    "G = nx.barbell_graph(m1=5, m2=3)\npos = nx.spiral_layout(G)\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}, {
    "name":
    "Multipartite Layout (分层布局)",
    "func":
    nx.multipartite_layout,
    "desc":
    """<p>将节点分层排列（通常是垂直或水平的直线层）。</p><ul><li><b>适用场景：</b>流程图、神经网络架构、或具有明确分级关系的数据。</li><li><b>特点：</b>需要为节点设置 <code>subset</code> 属性来分配层级。</li></ul>""",
    "code":
    "G = nx.barbell_graph(m1=5, m2=3)\nfor i, n in enumerate(G.nodes):\n    G.nodes[n]['subset'] = i % 3\npos = nx.multipartite_layout(G, subset_key='subset')\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}, {
    "name":
    "Planar Layout (平面布局)",
    "func":
    nx.planar_layout,
    "desc":
    """<p>尝试在不让边交叉的情况下绘制图。</p><ul><li><b>适用场景：</b>仅适用于<b>平面图</b> (Planar Graph)。如果图包含不可避免的交叉边，该函数会报错。</li><li><b>特点：</b>整洁，适合展示拓扑结构。</li></ul><p>注意：这里使用了一个小型的迷宫图作为示例，因为 Barbell Graph 不是平面图。</p>""",
    "code":
    "# 一个小型的迷宫图，结构清晰\nG = nx.sedgewick_maze_graph()\npos = nx.planar_layout(G)\nnx.draw(G, pos, with_labels=True, node_color='#FF4B4B', node_size=300,edge_color='#888888', font_color='white', font_size=10, ax=ax)"
}]

# -----------------------------------------------------------------------------
# 4. 主程序渲染
# -----------------------------------------------------------------------------

st.title("🕸️ NetworkX Drawing Layouts 指南")
st.markdown("为了保证每种布局都能得到体现，我们使用 **Barbell Graph (杠铃图)** 作为基础示例。")
st.markdown(r"""
<style>
    /* 容器样式：干净的白底卡片，带左侧强调色 */
    .doc-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            width: 500px;
    }

    /* 标题样式 */
    .doc-title {
        font-size: 1.3em;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* 描述文本 */
    .doc-text {
        color: #444;
        line-height: 1.6;
        margin-bottom: 15px;
    }

    /*不仅是加粗，增加一点背景色的强调文本 */
    .highlight {
        background-color: #f0f2f6;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        color: #d63384;
        font-size: 0.9em;
    }

    /* 分节标题 */
    .section-header {
        font-weight: 600;
        font-size: 1.1em;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }

    /* 列表样式优化 */
    .styled-list {
        list-style: none;
        padding-left: 0;
    }
    .styled-list li {
        margin-bottom: 8px;
        padding-left: 20px;
        position: relative;
    }
    .styled-list li::before {
        content: "•";
        color: #ff4b4b;
        font-weight: bold;
        position: absolute;
        left: 0;
    }

    /* 参数代码块 */
    .code-signature {
        background-color: #262730;
        color: #ffffff;
        padding: 12px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        margin-bottom: 15px;
        overflow-x: auto;
    }
</style>

<div class="doc-card">
    <div class="doc-title">🕸️ NetworkX Drawing Layouts 绘图布局示例</div>
    <div class="doc-text">
            <p>为了保证每种布局都能得到体现，我们使用<b>Barbell Graph (杠铃图)</b>作为基础示例。</p>
        <p>
            <code>nx.barbell_graph</code> 是 NetworkX 中用于生成<b>“杠铃图”</b>的生成器函数。
            它的名字来源于其拓扑结构：看起来像一个举重用的杠铃——<b>两头大（稠密），中间细（稀疏）</b>。
        </p>
    </div>
    <div class="section-header">1. 结构组成</div>
    <div class="doc-text">
        <ul class="styled-list">
            <li><b>🔴 左铃 (Left Bell)</b>：一个包含 m_1 个节点的<b>完全图</b>（所有节点互连）。</li>
            <li><b>🔴 右铃 (Right Bell)</b>：另一个包含 m_1 个节点的<b>完全图</b>。</li>
            <li><b>➖ 杆/手柄 (Handle/Bridge)</b>：一条包含 m_2 个节点的<b>路径图</b>，用于连接左铃和右铃。</li>
        </ul>
    </div>
    <div class="section-header">2. 函数参数</div>
    <div class="code-signature">
        G = nx.barbell_graph(m1, m2, create_using=None)
    </div>
    <div class="doc-text">
        <ul class="styled-list">
            <li>
                <span class="highlight">m1 (int)</span>：杠铃两端“铃”的节点数量。<br>
                <small style="color:#666">两端是完全图（Clique），所以两端各有 m_1 个节点。</small>
            </li>
            <li>
                <span class="highlight">m2 (int)</span>：中间连接“杆”的节点数量。<br>
                <small style="color:#666">如果是 <code>0</code>，则两端的完全图通过一条边直接相连。<br>
                如果是 <code>>0</code>，则中间会有 m_2 个节点排成一条线。</small>
            </li>
        </ul>
    </div>
</div>
""",
            unsafe_allow_html=True)
st.divider()

G = nx.barbell_graph(m1=5, m2=3)

COLS = 3
cols = st.columns(COLS)

for i, item in enumerate(layout_data):
    col_idx = i % COLS

    img_src, error_msg = plot_to_base64(G, item['func'])
    if item['func'] == nx.planar_layout:
        G = nx.sedgewick_maze_graph()
        img_src, error_msg = plot_to_base64(G, item['func'])

    if error_msg:
        img_html = f"<div style='height:100%; display:flex; align-items:center; justify-content:center; color:red; text-align:center'>无法绘制此布局<br><small>{error_msg}</small></div>"
    else:
        img_html = f"<img src='{img_src}' alt='Graph Plot'>"

    # --- 核心修正 ---
    # 拼接 HTML 时，不要在标签前加缩进（空格）。
    # 直接一行行拼，或者紧贴左边，避免 Markdown 将其识别为 Code Block。

    card_html = f"""
<div class="layout-card">
<div class="card-header"><p class="card-title">{item['name']}</p></div>
<div class="img-container">{img_html}</div>
<div class="scroll-content">
<div class="desc-text">{item['desc']}</div>
<div class="code-block">{item['code']}</div>
</div>
</div>
"""

    with cols[col_idx]:
        st.markdown(card_html, unsafe_allow_html=True)
