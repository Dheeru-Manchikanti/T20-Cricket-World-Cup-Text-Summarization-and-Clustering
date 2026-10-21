"""T20 Cricket World Cup News Analysis Using Text Summarization and Clustering.

A college-level NLP and Machine Learning Streamlit Web Application.
"""

from pathlib import Path
from typing import Dict, List, Optional
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dataset_loader import (
    DEFAULT_DATA_PATH,
    FALLBACK_DATA_PATH,
    get_dataset_statistics,
    load_dataset,
    validate_and_clean_dataset,
)
from src.preprocessing import clean_text, tokenize_words
from src.summarizer import summarize_text
from src.clustering import CricketNewsClusterer, compute_elbow_curve

# Output path for exported clustering results
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "clustered_news.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="T20 World Cup News Analyzer | NLP & ML",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme CSS
st.markdown(
    """
    <style>
    /* Global Typography & Palette */
    :root {
        --primary-cricket: #0f5132;
        --secondary-gold: #d4af37;
        --dark-navy: #0b192c;
        --card-bg: #f8f9fa;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0b192c;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #495057;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef2f7 100%);
        border: 1px solid #dbe2ea;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f5132;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-category {
        display: inline-block;
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-cluster {
        display: inline-block;
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .summary-box {
        background-color: #f1f8e9;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 6px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .original-box {
        background-color: #f8f9fa;
        border-left: 5px solid #607d8b;
        padding: 15px;
        border-radius: 6px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .highlight-term {
        background-color: #fff3cd;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Caching Data & Model
# ---------------------------------------------------------
@st.cache_data
def get_dataset(uploaded_file=None) -> pd.DataFrame:
    """Load and validate dataset with caching."""
    if uploaded_file is not None:
        return load_dataset(uploaded_file)
    return load_dataset()


@st.cache_resource
def get_trained_clusterer(texts: tuple, n_clusters: int) -> CricketNewsClusterer:
    """Fit and cache K-Means and TF-IDF models."""
    clusterer = CricketNewsClusterer(n_clusters=n_clusters, random_state=42)
    clusterer.fit(list(texts))
    return clusterer


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/cricket.png",
    width=64,
)
st.sidebar.title("T20 News NLP Hub")
st.sidebar.caption("Extractive Summarization & Topic Clustering")

# Navigation
page = st.sidebar.radio(
    "Go to Navigation",
    [
        "🏠 Home Dashboard",
        "🔍 News Explorer",
        "📝 Text Summarization",
        "📊 News Clustering",
        "⚡ Custom Article Analyzer",
        "📚 Dataset & Methodology",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Dataset & Model Settings")

# Dataset Source Selection
data_option = st.sidebar.selectbox(
    "Select Data Source",
    ["Curated T20 World Cup Dataset (107 Articles)", "Upload Custom CSV File"],
)

uploaded_file = None
if data_option == "Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (Columns: title, article_text / text)", type=["csv"]
    )

# Load data safely
try:
    df = get_dataset(uploaded_file)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Interactive Hyperparameters
n_clusters = st.sidebar.slider(
    "Number of Clusters (k)",
    min_value=2,
    max_value=min(8, max(2, len(df) - 1)),
    value=5,
    help="Configures the number of K-Means topic clusters.",
)

summary_sentence_limit = st.sidebar.slider(
    "Summary Sentences",
    min_value=2,
    max_value=5,
    value=3,
    help="Target number of sentences to extract for article summaries.",
)

# Fit clustering model (cached by text tuple and k)
text_tuple = tuple(df["article_text"].tolist())
clusterer = get_trained_clusterer(text_tuple, n_clusters=n_clusters)

# Enrich dataframe with cluster predictions
cluster_ids, cluster_labels = clusterer.get_cluster_assignments(df["article_text"].tolist())
df["cluster_id"] = cluster_ids
df["cluster_label"] = cluster_labels

# Sidebar Footer Stats
st.sidebar.divider()
st.sidebar.markdown(f"**Loaded Articles:** `{len(df)}`")
st.sidebar.markdown(f"**Fitted Clusters:** `{clusterer.n_clusters}`")
if clusterer.silhouette_avg is not None:
    st.sidebar.markdown(f"**Silhouette Score:** `{clusterer.silhouette_avg}`")


# ---------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# ---------------------------------------------------------
if page == "🏠 Home Dashboard":
    st.markdown('<div class="main-title">🏏 T20 Cricket World Cup News Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Automated NLP text summarization and unsupervised topic clustering '
        'for international T20 cricket news and tournament coverage.</div>',
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Cards
    stats = get_dataset_statistics(df)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{stats['total_articles']}</div>
                <div class="metric-lbl">Total Articles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{clusterer.n_clusters}</div>
                <div class="metric-lbl">Inferred Clusters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{stats['unique_categories']}</div>
                <div class="metric-lbl">News Categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        sil_display = f"{clusterer.silhouette_avg:.3f}" if clusterer.silhouette_avg is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{sil_display}</div>
                <div class="metric-lbl">Silhouette Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Visualizations Row 1: Cluster distribution & Category distribution
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("📊 Articles per Inferred Cluster")
        cluster_counts = df["cluster_label"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig_clusters = px.bar(
            cluster_counts,
            x="Count",
            y="Cluster",
            orientation="h",
            color="Count",
            color_continuous_scale="Greens",
            text="Count",
        )
        fig_clusters.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

    with row1_col2:
        st.subheader("🏷️ Distribution Across News Categories")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Articles"]
        fig_cats = px.pie(
            cat_counts,
            names="Category",
            values="Articles",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_cats.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_cats, use_container_width=True)

    st.markdown("---")

    # Visualizations Row 2: Top Cricket Vocabulary in Corpus
    st.subheader("🔤 Most Prominent Cricket Terminology in Corpus (TF-IDF Weight)")
    feature_names = clusterer.feature_names
    if feature_names and clusterer.vectorizer is not None:
        # Sum of TF-IDF scores across all documents
        tfidf_mat = clusterer.vectorizer.transform([clean_text(t) for t in df["article_text"]])
        mean_weights = np.asarray(tfidf_mat.mean(axis=0)).flatten()
        top_indices = mean_weights.argsort()[::-1][:15]

        top_words_df = pd.DataFrame(
            {
                "Term": [feature_names[i] for i in top_indices],
                "Mean TF-IDF Weight": [round(float(mean_weights[i]), 4) for i in top_indices],
            }
        ).sort_values("Mean TF-IDF Weight", ascending=True)

        fig_vocab = px.bar(
            top_words_df,
            x="Mean TF-IDF Weight",
            y="Term",
            orientation="h",
            color="Mean TF-IDF Weight",
            color_continuous_scale="Viridis",
            text="Mean TF-IDF Weight",
        )
        fig_vocab.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_vocab, use_container_width=True)

    # Recent Articles Feed
    st.subheader("📰 Recent Articles Snapshot")
    for _, row in df.head(3).iterrows():
        with st.container():
            st.markdown(f"#### {row['title']}")
            st.markdown(
                f"<span class='badge-category'>{row['category']}</span> "
                f"<span class='badge-cluster'>{row['cluster_label']}</span> "
                f"<span style='color: #6c757d; font-size: 0.85rem;'>Source: {row['source']} | Date: {row['published_date']}</span>",
                unsafe_allow_html=True,
            )
            preview = " ".join(row["article_text"].split()[:35]) + "..."
            st.write(preview)
            st.divider()


# ---------------------------------------------------------
# PAGE 2: NEWS EXPLORER
# ---------------------------------------------------------
elif page == "🔍 News Explorer":
    st.markdown('<div class="main-title">🔍 Cricket News Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Search, filter, and inspect articles across teams, topics, dates, and clusters.</div>',
        unsafe_allow_html=True,
    )

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        search_query = st.text_input("Search keywords in title or content:", placeholder="e.g., Bumrah, final, USA, wicket")
    with f_col2:
        cat_filter = st.selectbox(
            "Filter by Category:", ["All Categories"] + sorted(df["category"].unique().tolist())
        )
    with f_col3:
        cluster_filter = st.selectbox(
            "Filter by Cluster:", ["All Clusters"] + sorted(df["cluster_label"].unique().tolist())
        )

    # Apply Filters
    filtered_df = df.copy()
    if search_query:
        q = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["title"].str.lower().str.contains(q, na=False)
            | filtered_df["article_text"].str.lower().str.contains(q, na=False)
        ]
    if cat_filter != "All Categories":
        filtered_df = filtered_df[filtered_df["category"] == cat_filter]
    if cluster_filter != "All Clusters":
        filtered_df = filtered_df[filtered_df["cluster_label"] == cluster_filter]

    st.write(f"Showing **{len(filtered_df)}** matching articles:")

    # Table Display
    display_cols = ["article_id", "title", "category", "source", "published_date", "cluster_label"]
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Detailed Article Reader
    st.subheader("📖 Read Full Article")
    if not filtered_df.empty:
        article_titles = filtered_df["title"].tolist()
        selected_title = st.selectbox("Select an article to read:", article_titles)
        selected_article = filtered_df[filtered_df["title"] == selected_title].iloc[0]

        st.markdown(f"### {selected_article['title']}")
        st.markdown(
            f"<span class='badge-category'>{selected_article['category']}</span> "
            f"<span class='badge-cluster'>{selected_article['cluster_label']}</span> "
            f"<span style='color: #6c757d; font-size: 0.85rem;'>Source: {selected_article['source']} | ID: {selected_article['article_id']} | Date: {selected_article['published_date']}</span>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="original-box" style="margin-top: 15px;">', unsafe_allow_html=True)
        st.write(selected_article["article_text"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick action button to trigger summarizer
        if st.button("🚀 Summarize this Article Now"):
            st.session_state["summarize_article_title"] = selected_article["title"]
            st.success("Article sent to Text Summarization tab! Click '📝 Text Summarization' in sidebar.")
    else:
        st.info("No articles match the current filter criteria.")


# ---------------------------------------------------------
# PAGE 3: TEXT SUMMARIZATION
# ---------------------------------------------------------
elif page == "📝 Text Summarization":
    st.markdown('<div class="main-title">📝 Extractive Text Summarization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Generate concise summaries using frequency-based sentence ranking, '
        'lead bias, cricket statistic preservation, and Jaccard deduplication.</div>',
        unsafe_allow_html=True,
    )

    # Input Method: Dataset vs Custom Input
    input_mode = st.radio("Choose Input Mode:", ["Select Existing Article from Dataset", "Paste Custom Cricket Text"], horizontal=True)

    article_title = ""
    raw_article_text = ""

    if input_mode == "Select Existing Article from Dataset":
        default_index = 0
        if "summarize_article_title" in st.session_state:
            matching_indices = df.index[df["title"] == st.session_state["summarize_article_title"]].tolist()
            if matching_indices:
                default_index = matching_indices[0]

        selected_art_idx = st.selectbox(
            "Select Article:",
            options=range(len(df)),
            format_func=lambda idx: f"{df.iloc[idx]['title']} ({df.iloc[idx]['category']})",
            index=default_index,
        )
        selected_row = df.iloc[selected_art_idx]
        article_title = selected_row["title"]
        raw_article_text = selected_row["article_text"]
    else:
        article_title = st.text_input("Article Title (optional):", value="Custom T20 Match Analysis")
        raw_article_text = st.text_area(
            "Paste Cricket News Article Text:",
            height=200,
            placeholder="Paste cricket match report, squad analysis, or tournament news here...",
        )

    # Sentence Count Slider
    max_sents = st.slider(
        "Target Summary Length (Sentences):",
        min_value=1,
        max_value=5,
        value=summary_sentence_limit,
    )

    if st.button("✨ Generate Extractive Summary", type="primary"):
        if not raw_article_text or len(raw_article_text.strip()) == 0:
            st.warning("Please provide or select article text to summarize.")
        else:
            with st.spinner("Analyzing sentence scores and extracting key insights..."):
                summary_result = summarize_text(raw_article_text, max_sentences=max_sents)

            # Metrics Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Original Words", summary_result["original_word_count"])
            with m_col2:
                st.metric("Summary Words", summary_result["summary_word_count"])
            with m_col3:
                st.metric("Word Reduction", f"-{summary_result['reduction_percentage']}%")
            with m_col4:
                st.metric(
                    "Sentence Ratio",
                    f"{summary_result['summary_sentences']} / {summary_result['original_sentences']}",
                )

            st.markdown("---")

            # Side-by-Side Comparison
            col_orig, col_summ = st.columns(2)

            with col_orig:
                st.subheader("📄 Original Full Text")
                st.markdown(f"**{article_title}**")
                st.markdown('<div class="original-box">', unsafe_allow_html=True)
                st.write(raw_article_text)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_summ:
                st.subheader("🎯 Generated Extractive Summary")
                st.markdown(f"**{article_title}** (Summary)")
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.write(summary_result["summary"])
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("##### 📋 Copy-Friendly Output:")
                st.code(summary_result["summary"], language="text")

            # Explanation of Extractive Integrity
            st.info(
                "💡 **Extractive NLP Integrity:** Extractive summarization scores and extracts the most information-dense "
                "sentences directly from the source text without generative hallucination. Important factual scores, "
                "player names, and match results are preserved exactly as reported."
            )


# ---------------------------------------------------------
# PAGE 4: NEWS CLUSTERING
# ---------------------------------------------------------
elif page == "📊 News Clustering":
    st.markdown('<div class="main-title">📊 Unsupervised News Clustering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Discover latent topics and group cricket news using TF-IDF feature matrices '
        'and K-Means clustering.</div>',
        unsafe_allow_html=True,
    )

    # Cluster Cards Overview
    st.subheader(f"📌 Discovered Clusters (k = {clusterer.n_clusters})")
    representatives = clusterer.get_representative_articles(df, top_per_cluster=2)

    cols = st.columns(min(clusterer.n_clusters, 4))
    for c_id in range(clusterer.n_clusters):
        col_target = cols[c_id % len(cols)]
        with col_target:
            cluster_count = (df["cluster_id"] == c_id).sum()
            keywords = clusterer.cluster_keywords.get(c_id, [])
            kw_str = ", ".join(keywords[:4])

            with st.container():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ced4da; border-radius: 8px; padding: 12px; margin-bottom: 12px; background: #fff;">
                        <h4 style="margin: 0; color: #0f5132;">Cluster {c_id + 1}</h4>
                        <div style="font-size: 0.8rem; color: #6c757d; margin-bottom: 6px;">Articles: <b>{cluster_count}</b></div>
                        <div style="font-size: 0.85rem;"><b>Keywords:</b><br/>{kw_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # 2D SVD Cluster Visualization
    st.subheader("🗺️ 2D Topic Cluster Visualization (TruncatedSVD Projection)")
    st.caption("Documents projected into 2D space from high-dimensional TF-IDF vectors using TruncatedSVD.")

    coords_2d = clusterer.transform_2d(df["article_text"].tolist())
    plot_df = pd.DataFrame(
        {
            "SVD Dimension 1": coords_2d[:, 0],
            "SVD Dimension 2": coords_2d[:, 1],
            "Cluster": df["cluster_label"],
            "Title": df["title"],
            "Category": df["category"],
            "Source": df["source"],
        }
    )

    fig_svd = px.scatter(
        plot_df,
        x="SVD Dimension 1",
        y="SVD Dimension 2",
        color="Cluster",
        hover_data=["Title", "Category", "Source"],
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_svd.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")))
    fig_svd.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_svd, use_container_width=True)

    st.markdown("---")

    # Cluster Prototypes & Representative Articles
    st.subheader("🏆 Representative Articles per Cluster")
    st.caption("Articles with the minimum Euclidean distance to their respective cluster centroids.")

    for c_id in range(clusterer.n_clusters):
        c_label = clusterer.cluster_labels.get(c_id, f"Cluster {c_id + 1}")
        c_kws = ", ".join(clusterer.cluster_keywords.get(c_id, []))
        c_subset = df[df["cluster_id"] == c_id]

        with st.expander(f"📁 {c_label} — ({len(c_subset)} total articles)", expanded=False):
            st.markdown(f"**Top Cluster Terms:** `{c_kws}`")
            rep_df = representatives.get(c_id, pd.DataFrame())

            if not rep_df.empty:
                st.markdown("**Prototypical Articles (Centroid Closest):**")
                for _, row in rep_df.iterrows():
                    st.markdown(f"- 📄 **{row['title']}** (*{row['source']}*, {row['published_date']})")
                    st.caption(" ".join(row["article_text"].split()[:30]) + "...")

            st.markdown("**All Articles in this Cluster:**")
            st.dataframe(
                c_subset[["article_id", "title", "category", "source", "published_date"]],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # Elbow Method and Silhouette Evaluation
    show_elbow = st.checkbox("📈 Show Elbow Method & Silhouette Evaluation Analysis", value=False)
    if show_elbow:
        st.subheader("📐 Model Validation: Elbow Method & Silhouette Analysis")
        with st.spinner("Computing Inertia and Silhouette across k=2..8..."):
            elbow_data = compute_elbow_curve(df["article_text"].tolist(), max_k=8)

        if elbow_data["k_values"]:
            el_col1, el_col2 = st.columns(2)
            with el_col1:
                fig_inertia = px.line(
                    x=elbow_data["k_values"],
                    y=elbow_data["inertias"],
                    markers=True,
                    labels={"x": "Number of Clusters (k)", "y": "Sum of Squared Distances (Inertia)"},
                    title="Elbow Curve (Inertia vs k)",
                )
                fig_inertia.update_traces(line_color="#0f5132", line_width=3)
                st.plotly_chart(fig_inertia, use_container_width=True)

            with el_col2:
                fig_sil = px.line(
                    x=elbow_data["k_values"],
                    y=elbow_data["silhouettes"],
                    markers=True,
                    labels={"x": "Number of Clusters (k)", "y": "Average Silhouette Score"},
                    title="Silhouette Score vs k",
                )
                fig_sil.update_traces(line_color="#d4af37", line_width=3)
                st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    # Export & Download Clustered Dataset
    st.subheader("💾 Export Clustered Dataset")
    export_df = df[["article_id", "title", "category", "source", "published_date", "cluster_id", "cluster_label", "article_text"]].copy()

    e_col1, e_col2 = st.columns(2)
    with e_col1:
        if st.button("Save Clustered Dataset to outputs/clustered_news.csv"):
            export_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
            st.success(f"Saved {len(export_df)} clustered articles to {OUTPUT_FILE}!")

    with e_col2:
        csv_buffer = io.BytesIO()
        export_df.to_csv(csv_buffer, index=False, encoding="utf-8")
        st.download_button(
            label="📥 Download Clustered CSV File",
            data=csv_buffer.getvalue(),
            file_name="t20_clustered_news.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# PAGE 5: CUSTOM ARTICLE ANALYZER
# ---------------------------------------------------------
elif page == "⚡ Custom Article Analyzer":
    st.markdown('<div class="main-title">⚡ Custom Cricket Article Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Paste any real-world cricket news report to generate an instant extractive summary '
        'and assign it to the nearest fitted K-Means topic cluster.</div>',
        unsafe_allow_html=True,
    )

    # Preset Examples
    st.write("**Quick-fill preset examples:**")
    p1, p2, p3 = st.columns(3)

    preset_title = ""
    preset_text = ""

    if p1.button("🏆 2024 Final Match Report"):
        preset_title = "India Crowned T20 World Champions in Thrilling Barbados Finish"
        preset_text = (
            "India secured their second T20 World Cup title with a nail-biting seven-run victory over South Africa "
            "at Kensington Oval in Barbados. Virat Kohli led the batting recovery with 76 runs off 59 balls, pushing India "
            "to 176 for 7. In response, Heinrich Klaasen threatened an upset with a blistering 52. However, Jasprit Bumrah's "
            "impenetrable death bowling and Hardik Pandya's composure in the final over sealed the championship for Rohit Sharma's side."
        )

    if p2.button("⚡ USA Super Over Upset"):
        preset_title = "USA Stun Pakistan in Epic Dallas Super Over Encounter"
        preset_text = (
            "The United States registered one of the greatest upsets in cricket history by overcoming Pakistan in a historic "
            "Super Over in Dallas. Monank Patel and Aaron Jones led the chase to tie the match on 159. In the Super Over, "
            "Saurabh Netravalkar defended 18 runs against Iftikhar Ahmed and Shadab Khan to spark jubilant scenes across America."
        )

    if p3.button("🎯 Pat Cummins Hat-Trick"):
        preset_title = "Pat Cummins Takes Back-to-Back World Cup Hat-Tricks"
        preset_text = (
            "Australia Test captain Pat Cummins made cricket history by taking consecutive hat-tricks against Bangladesh and "
            "Afghanistan in the Super Eights. Cummins dismissed star batters across multiple overs with pinpoint slower balls "
            "and yorkers, demonstrating masterclass death-bowling execution."
        )

    custom_title = st.text_input("Article Title:", value=preset_title, placeholder="Enter article headline...")
    custom_text = st.text_area(
        "Article Content:",
        value=preset_text,
        height=220,
        placeholder="Type or paste any cricket news article text here...",
    )

    custom_sentences = st.slider("Target Summary Sentences:", min_value=1, max_value=4, value=2)

    if st.button("🚀 Analyze Custom Article", type="primary"):
        if not custom_text.strip():
            st.warning("Please enter or paste article text to analyze.")
        else:
            with st.spinner("Processing NLP Pipeline (Cleaning -> Summarization -> Vectorization -> Cluster Prediction)..."):
                # 1. Summarization
                summary_info = summarize_text(custom_text, max_sentences=custom_sentences)

                # 2. Cluster Prediction
                prediction_info = clusterer.predict_custom_article(custom_text)

            st.success("Analysis Complete!")

            # Results Section
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.subheader("🎯 Extractive Summary")
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.write(summary_info["summary"])
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(
                    f"**Word Count:** Original: `{summary_info['original_word_count']}` | "
                    f"Summary: `{summary_info['summary_word_count']}` | "
                    f"Reduction: `-{summary_info['reduction_percentage']}%`"
                )

            with res_col2:
                st.subheader("🏷️ Predicted Topic Cluster")
                pred_c_id = prediction_info["predicted_cluster_id"]
                pred_label = prediction_info["cluster_label"]
                pred_kws = ", ".join(prediction_info["cluster_keywords"][:5])
                doc_kws = ", ".join(prediction_info["document_top_keywords"])

                st.markdown(
                    f"""
                    <div style="background-color: #e3f2fd; border-left: 5px solid #1976d2; padding: 15px; border-radius: 6px;">
                        <h4 style="margin: 0; color: #0d47a1;">Assigned to: {pred_label}</h4>
                        <p style="margin: 8px 0 0 0; font-size: 0.9rem;">
                            <b>Centroid Euclidean Distance:</b> <code>{prediction_info['centroid_distance']}</code><br/>
                            <b>Cluster Key Terms:</b> {pred_kws}<br/>
                            <b>Document Matched Terms:</b> {doc_kws}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.info(
                    "ℹ️ **Methodology Note:** The custom article is mapped to the TF-IDF feature space using the "
                    "pre-fitted vocabulary and assigned to the closest K-Means centroid based on Euclidean similarity. "
                    "This represents unsupervised text similarity rather than subjective editorial categorization."
                )


# ---------------------------------------------------------
# PAGE 6: DATASET & METHODOLOGY
# ---------------------------------------------------------
elif page == "📚 Dataset & Methodology":
    st.markdown('<div class="main-title">📚 Dataset & Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed documentation of project architecture, NLP algorithms, '
        'dataset provenance, and academic design.</div>',
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.tabs(["📊 Dataset Overview & Statistics", "⚙️ NLP & ML Architecture", "🎓 Academic Evaluation"])

    with t1:
        st.subheader("Dataset Provenance & Quality")
        st.markdown(
            """
            - **Dataset File:** `data/t20_cricket_news.csv`
            - **Total Records:** 107 richly detailed cricket articles.
            - **Tournament Coverage:** ICC Men's T20 World Cup tournaments (2007 through 2024), focusing on the historic 2024 edition in USA and West Indies.
            - **Authentic Attributions:** Articles are attributed to leading verified cricket sports news outlets:
              * *ICC Media Release*
              * *ESPNcricinfo Staff*
              * *BBC Sport Cricket*
              * *The Guardian Sport*
              * *Reuters Cricket*
              * *Cricbuzz Desk*
            - **Schema Details:**
              * `article_id`: Unique identifier (e.g. `T20-001`)
              * `title`: Clean headline
              * `article_text`: Full news article text
              * `category`: Editorial category
              * `source`: Reporting outlet
              * `published_date`: Event publication date
            """
        )

        st.markdown("##### Dataset Preview (Top 10 Records):")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("##### Word Count Distribution:")
        word_counts = df["article_text"].apply(lambda t: len(str(t).split()))
        fig_hist = px.histogram(
            word_counts,
            nbins=15,
            labels={"value": "Article Word Count", "count": "Number of Articles"},
            title="Word Count Histogram",
            color_discrete_sequence=["#0f5132"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with t2:
        st.subheader("System Architecture & Algorithms")
        st.markdown(
            """
            ### 1. Text Preprocessing Pipeline
            Raw text is cleaned through modular steps in `src/preprocessing.py`:
            - **HTML & URL Stripping:** Regex removal of HTML tags (`<[^>]+>`) and web links.
            - **Entity Preservation:** Specialized patterns protect cricket scores (`176/7`), bowling figures (`4-0-18-2`), and runs/wickets.
            - **Tokenization:** Sentence tokenization via NLTK `sent_tokenize` (with an active regex boundary fallback).
            - **Stopword Filtering:** NLTK English stopwords combined with a robust built-in fallback set.

            ### 2. Extractive Text Summarization
            Implemented in `src/summarizer.py`:
            - **Sentence Scoring:** Computes average normalized term frequencies for content words.
            - **Lead Bias:** Sports journalism places core outcomes in lead paragraphs; sentences receive position weighting: $W_{pos} = \\frac{1}{1 + 0.15 \\times index}$.
            - **Cricket Entity Boost:** 1.25x weight applied to sentences containing wickets, runs, or overs.
            - **Jaccard Deduplication:** Sentence overlap checked; sentences with Jaccard coefficient $> 0.65$ are skipped to avoid redundancy.
            - **Chronological Restructuring:** Selected sentences are sorted back to their natural document order.

            ### 3. Document Clustering (TF-IDF + K-Means)
            Implemented in `src/clustering.py`:
            - **TF-IDF Vectorization:** Sublinear term frequency scaling with unigram and bigram feature extraction ($max\\_features = 3000$).
            - **K-Means Clustering:** Minimizes within-cluster sum of squares (inertia) over 10 randomized restarts ($n\\_init = 10$).
            - **Topic Keyword Extraction:** Top feature indices sorted from cluster centroid coordinates.
            - **2D Dimensionality Reduction:** TruncatedSVD projects high-dimensional matrices into 2 principal components for visualization.
            """
        )

    with t3:
        st.subheader("Academic Evaluation, Limitations & Future Scope")
        st.markdown(
            """
            ### Project Merits & Strengths
            1. **Zero Hallucination:** Extractive summarization extracts verified sentences directly from the original report, guaranteeing numerical and factual accuracy for sports statistics.
            2. **Deterministic & Lightweight:** Runs completely locally without expensive API keys or heavy GPU requirements.
            3. **Resilience & Fallbacks:** Operates seamlessly even if NLTK download servers are unavailable.

            ### Known Limitations
            1. **Extractive Boundaries:** Extractive summaries cannot rephrase or compress sentences into novel abstract prose.
            2. **Unsupervised Topic Ambiguity:** K-Means cluster labels are derived from term frequencies and require human interpretation to align with editorial taxonomy.

            ### Future Enhancements
            1. Integration of abstractive transformer models (e.g. BART or T5) fine-tuned on cricket domain corpora.
            2. Named Entity Recognition (NER) for automated extraction of player of the match, venue, and run rates.
            3. Dynamic live RSS ingestion for real-time tournament feed tracking.
            """
        )

# Footer
st.markdown("---")
st.caption(
    "T20 Cricket World Cup News Analysis | Built with Streamlit, Scikit-learn, NLTK, Pandas, and Plotly."
)
