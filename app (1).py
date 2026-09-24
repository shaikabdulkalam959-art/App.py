"""
Sai University — Research Intelligence Dashboard
==================================================
An interactive Streamlit dashboard that consumes the Sai University
Publications API and presents research output, quality, institutional
contribution, and impact metrics for university leadership, faculty,
and researchers.

Data source: https://sai-publications-dashboard.vercel.app/api/publications

Run with:
    streamlit run app.py
"""

import io
from collections import Counter

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --------------------------------------------------------------------------
# Page config & global style
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sai University — Research Intelligence Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = ["#0B3D91", "#1E88E5", "#26A69A", "#F4B400", "#DB4437",
           "#8E24AA", "#00897B", "#6D4C41", "#546E7A", "#C0CA33"]
sns.set_palette(PALETTE)

API_URL = "https://sai-publications-dashboard.vercel.app/api/publications"

SDG_NAMES = {
    "1": "No Poverty", "2": "Zero Hunger", "3": "Good Health & Well-being",
    "4": "Quality Education", "5": "Gender Equality",
    "6": "Clean Water & Sanitation", "7": "Affordable & Clean Energy",
    "8": "Decent Work & Economic Growth",
    "9": "Industry, Innovation & Infrastructure",
    "10": "Reduced Inequalities", "11": "Sustainable Cities & Communities",
    "12": "Responsible Consumption & Production", "13": "Climate Action",
    "14": "Life Below Water", "15": "Life on Land",
    "16": "Peace, Justice & Strong Institutions",
    "17": "Partnerships for the Goals",
}

# --------------------------------------------------------------------------
# Data loading & cleaning
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching publication records...")
def fetch_raw():
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    return pd.DataFrame(records)


def _split_pipe(val):
    """Split a '|'-delimited field into a clean list, dropping blanks."""
    if pd.isna(val) or str(val).strip() in ("", "-"):
        return []
    return [p.strip() for p in str(val).split("|") if p.strip()]


def _split_indexing(val):
    if pd.isna(val) or str(val).strip() in ("", "-"):
        return ["Non Indexed"]
    return [p.strip() for p in str(val).split("|") if p.strip()]


@st.cache_data(ttl=3600, show_spinner="Cleaning & structuring data...")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalise blanks to NaN across the board
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.replace("-", np.nan)

    # Year: numeric, drop unparsable
    df["Year"] = pd.to_numeric(df.get("Year"), errors="coerce")
    df = df[df["Year"].notna()].copy()
    df["Year"] = df["Year"].astype(int)

    # Quartile: prefer SJR Quartile, fall back to Year Wise Quartile
    df["Quartile"] = df.get("SJR Quartile")
    if "Year Wise Quartile" in df.columns:
        df["Quartile"] = df["Quartile"].fillna(df["Year Wise Quartile"])
    df["Quartile"] = df["Quartile"].where(
        df["Quartile"].isin(["Q1", "Q2", "Q3", "Q4"]), np.nan
    )

    # Indexing status -> list of {Scopus, WoS, Non Indexed, ...}
    df["Indexing List"] = df.get("Indexing Status").apply(_split_indexing)
    df["Is Scopus"] = df["Indexing List"].apply(lambda l: "Scopus" in l)
    df["Is WoS"] = df["Indexing List"].apply(lambda l: "WoS" in l)
    df["Is Indexed"] = df["Indexing List"].apply(
        lambda l: any(x != "Non Indexed" for x in l)
    )

    # SDGs: prefer SaiU SDG Indexing, fall back to Scopus/WoS achieved SDGs
    def combined_sdgs(row):
        for col in ("SaiU SDG Indexing", "Achieved SDG (Scopus)", "Achieved SDG (WoS)"):
            vals = _split_pipe(row.get(col))
            if vals:
                return sorted(set(v for v in vals if v.isdigit()))
        return []

    df["SDG List"] = df.apply(combined_sdgs, axis=1)

    # Authors: primary SaiU-affiliated author(s) driving the record
    df["SaiU Authors"] = df.get("SaiU Authors", "").fillna("")
    df["SaiU Author List"] = df["SaiU Authors"].apply(
        lambda v: [a.strip() for a in str(v).split(",") if a.strip()]
    )
    df["Primary SaiU Author"] = df["SaiU Author List"].apply(
        lambda l: l[0] if l else "Unknown"
    )

    df["School"] = df.get("School").fillna("Unspecified")
    df["Document Type"] = df.get("Document Type").fillna("Unspecified")
    df["Publisher"] = df.get("Publisher").fillna("Unknown")
    df["Source title"] = df.get("Source title").fillna("Unknown")
    df["Title"] = df.get("Title").fillna("Untitled")

    # A best-effort unique key to de-duplicate a publication that is
    # repeated once per co-authoring SaiU faculty member
    df["DOI"] = df.get("DOI")
    df["Pub Key"] = df["DOI"].fillna(
        df["Title"].str.lower().str.strip() + "||" + df["Year"].astype(str)
    )

    # Best available link for the explorer
    def best_link(row):
        for col in ("DOI Link", "Article Link", "Journal Link", "Scopus URL", "WoS URL"):
            v = row.get(col)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    df["Best Link"] = df.apply(best_link, axis=1)

    return df.reset_index(drop=True)


def dedupe_publications(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse author-level rows to one row per unique publication."""
    if df.empty:
        return df
    return df.sort_values("Year").drop_duplicates(subset="Pub Key", keep="first")


def explode_col(df: pd.DataFrame, list_col: str, out_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[out_name])
    return df[[list_col]].explode(list_col).rename(columns={list_col: out_name}).dropna()


# --------------------------------------------------------------------------
# Load & clean
# --------------------------------------------------------------------------
try:
    raw_df = fetch_raw()
except Exception as e:
    st.error(f"Could not fetch publication data from the API: {e}")
    st.stop()

df_all = clean_data(raw_df)

if df_all.empty:
    st.warning("No usable publication records were returned by the API.")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar — Filters
# --------------------------------------------------------------------------
st.sidebar.title("🎛️ Filters")
st.sidebar.caption("All KPIs, charts and the explorer respond to these filters.")

year_min, year_max = int(df_all["Year"].min()), int(df_all["Year"].max())
year_range = st.sidebar.slider(
    "Publication Year", min_value=year_min, max_value=year_max,
    value=(year_min, year_max),
)

schools = sorted(df_all["School"].dropna().unique().tolist())
sel_schools = st.sidebar.multiselect("School", schools, default=[])

authors = sorted(a for a in df_all["SaiU Author List"].explode().dropna().unique() if a)
sel_authors = st.sidebar.multiselect("Author (SaiU faculty)", authors, default=[])

doc_types = sorted(df_all["Document Type"].dropna().unique().tolist())
sel_doctypes = st.sidebar.multiselect("Document Type", doc_types, default=[])

all_indexing = sorted(set(x for lst in df_all["Indexing List"] for x in lst))
sel_indexing = st.sidebar.multiselect("Indexing Status", all_indexing, default=[])

quartiles = ["Q1", "Q2", "Q3", "Q4"]
sel_quartiles = st.sidebar.multiselect("SJR Quartile", quartiles, default=[])

all_sdgs = sorted(set(x for lst in df_all["SDG List"] for x in lst), key=int)
sdg_options = [f"SDG {n} — {SDG_NAMES.get(n, n)}" for n in all_sdgs]
sel_sdg_labels = st.sidebar.multiselect("SDG", sdg_options, default=[])
sel_sdgs = [s.split(" ")[1] for s in sel_sdg_labels]

pub_query = st.sidebar.text_input("Publisher / Source contains")

st.sidebar.markdown("---")
if st.sidebar.button("↺ Reset filters"):
    st.rerun()

# --------------------------------------------------------------------------
# Apply filters (on the author-record level table)
# --------------------------------------------------------------------------
mask = df_all["Year"].between(year_range[0], year_range[1])
if sel_schools:
    mask &= df_all["School"].isin(sel_schools)
if sel_authors:
    mask &= df_all["SaiU Author List"].apply(lambda l: any(a in sel_authors for a in l))
if sel_doctypes:
    mask &= df_all["Document Type"].isin(sel_doctypes)
if sel_indexing:
    mask &= df_all["Indexing List"].apply(lambda l: any(i in sel_indexing for i in l))
if sel_quartiles:
    mask &= df_all["Quartile"].isin(sel_quartiles)
if sel_sdgs:
    mask &= df_all["SDG List"].apply(lambda l: any(s in sel_sdgs for s in l))
if pub_query:
    q = pub_query.lower()
    mask &= (
        df_all["Publisher"].str.lower().str.contains(q, na=False)
        | df_all["Source title"].str.lower().str.contains(q, na=False)
    )

df = df_all[mask].copy()
df_pubs = dedupe_publications(df)  # one row per unique publication

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🎓 Sai University — Research Intelligence Dashboard")
st.caption(
    "Explore research output, publication quality, institutional contribution, "
    "and research impact across schools, faculty and years."
)

if df.empty:
    st.warning("No publications match the current filters. Try widening your selection.")
    st.stop()

# --------------------------------------------------------------------------
# KPI Section
# --------------------------------------------------------------------------
st.subheader("📊 Key Performance Indicators")

total_pubs = df_pubs["Pub Key"].nunique()
n_years = df_pubs["Year"].nunique()
n_scopus = df_pubs["Is Scopus"].sum()
n_wos = df_pubs["Is WoS"].sum()
n_q1 = (df_pubs["Quartile"] == "Q1").sum()
n_schools = df["School"].nunique()
n_faculty = df["SaiU Author List"].explode().dropna().nunique()
n_sdgs = len(set(x for lst in df_pubs["SDG List"] for x in lst))
pct_indexed = 100 * df_pubs["Is Indexed"].mean() if total_pubs else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Publications", f"{total_pubs:,}")
k2.metric("Years Covered", f"{n_years}", f"{year_range[0]}–{year_range[1]}")
k3.metric("Scopus-Indexed", f"{int(n_scopus):,}", f"{100*n_scopus/total_pubs:.0f}%" if total_pubs else "0%")
k4.metric("Web of Science-Indexed", f"{int(n_wos):,}", f"{100*n_wos/total_pubs:.0f}%" if total_pubs else "0%")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Q1 Publications", f"{int(n_q1):,}", f"{100*n_q1/total_pubs:.0f}%" if total_pubs else "0%")
k6.metric("Schools Represented", f"{n_schools}")
k7.metric("Faculty Authors", f"{n_faculty}")
k8.metric("SDGs Represented", f"{n_sdgs} / 17")

st.caption(f"Overall indexed share (Scopus and/or WoS): **{pct_indexed:.1f}%**")
st.markdown("---")

# --------------------------------------------------------------------------
# Auto-generated insights
# --------------------------------------------------------------------------
with st.expander("💡 Automatically generated insights", expanded=True):
    insights = []
    by_year = df_pubs.groupby("Year")["Pub Key"].nunique()
    if len(by_year) >= 2:
        yoy = by_year.pct_change().iloc[-1] * 100
        trend_word = "grew" if yoy >= 0 else "declined"
        insights.append(
            f"Publication output **{trend_word} {abs(yoy):.0f}%** from "
            f"{by_year.index[-2]} to {by_year.index[-1]} "
            f"({int(by_year.iloc[-2])} → {int(by_year.iloc[-1])})."
        )
    top_school = df.groupby("School")["Pub Key"].nunique().idxmax()
    top_school_n = df.groupby("School")["Pub Key"].nunique().max()
    insights.append(f"**{top_school}** leads output with **{top_school_n}** publications in the current view.")
    top_author_counts = df.explode("SaiU Author List").groupby("SaiU Author List")["Pub Key"].nunique()
    if not top_author_counts.empty:
        top_author = top_author_counts.idxmax()
        insights.append(f"**{top_author}** is the most prolific author shown, with **{int(top_author_counts.max())}** publications.")
    if total_pubs:
        insights.append(f"**{pct_indexed:.0f}%** of publications carry Scopus and/or WoS indexing; **{100*n_q1/total_pubs:.0f}%** land in Q1 journals.")
    sdg_counts = Counter(x for lst in df_pubs["SDG List"] for x in lst)
    if sdg_counts:
        top_sdg, top_sdg_n = sdg_counts.most_common(1)[0]
        insights.append(f"The most represented SDG is **SDG {top_sdg} — {SDG_NAMES.get(top_sdg, top_sdg)}** ({top_sdg_n} publications).")
    for line in insights:
        st.markdown(f"- {line}")

# --------------------------------------------------------------------------
# Visualizations
# --------------------------------------------------------------------------
st.subheader("📈 Visualizations")
tab_trend, tab_school, tab_quality, tab_people, tab_sdg = st.tabs(
    ["Trends", "Schools & Types", "Quality & Indexing", "Faculty Contribution", "SDG Alignment"]
)

# --- Trends over time -------------------------------------------------
with tab_trend:
    c1, c2 = st.columns([2, 1])
    with c1:
        yearly = df_pubs.groupby("Year")["Pub Key"].nunique().reindex(
            range(year_range[0], year_range[1] + 1), fill_value=0
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(yearly.index, yearly.values, marker="o", color=PALETTE[0], linewidth=2.5)
        ax.fill_between(yearly.index, yearly.values, alpha=0.15, color=PALETTE[0])
        ax.set_title("Publication Output Over Time")
        ax.set_xlabel("Year"); ax.set_ylabel("Publications")
        ax.set_xticks(list(yearly.index))
        fig.tight_layout()
        st.pyplot(fig)
    with c2:
        st.markdown("**Year-over-year table**")
        yoy_df = yearly.to_frame("Publications")
        yoy_df["Change"] = yoy_df["Publications"].diff()
        st.dataframe(yoy_df, use_container_width=True)

    # Indexed vs non-indexed trend
    trend_idx = df_pubs.groupby(["Year", "Is Indexed"])["Pub Key"].nunique().unstack(fill_value=0)
    trend_idx = trend_idx.reindex(range(year_range[0], year_range[1] + 1), fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(trend_idx))
    labels = {False: "Non-indexed", True: "Indexed (Scopus/WoS)"}
    for col in [False, True]:
        if col in trend_idx.columns:
            ax2.bar(trend_idx.index.astype(str), trend_idx[col], bottom=bottom,
                    label=labels[col], color=PALETTE[1] if col else PALETTE[8])
            bottom += trend_idx[col].values
    ax2.set_title("Indexed vs. Non-Indexed Publications by Year")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Publications")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)

# --- Schools & document types ------------------------------------------
with tab_school:
    c1, c2 = st.columns(2)
    with c1:
        school_counts = df.groupby("School")["Pub Key"].nunique().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(school_counts))))
        ax.barh(school_counts.index, school_counts.values, color=PALETTE[2])
        ax.set_title("Publications by School")
        ax.set_xlabel("Publications")
        fig.tight_layout()
        st.pyplot(fig)
    with c2:
        dt_counts = df_pubs["Document Type"].value_counts()
        fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(dt_counts))))
        ax.pie(dt_counts.values, labels=dt_counts.index, autopct="%1.0f%%",
               colors=PALETTE, wedgeprops={"edgecolor": "white"})
        ax.set_title("Publications by Document Type")
        fig.tight_layout()
        st.pyplot(fig)

    # School x Year heatmap (bonus)
    st.markdown("**School activity over time**")
    heat = df.groupby(["School", "Year"])["Pub Key"].nunique().unstack(fill_value=0)
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(heat))))
        sns.heatmap(heat, cmap="Blues", annot=True, fmt="d", cbar_kws={"label": "Publications"}, ax=ax)
        ax.set_xlabel("Year"); ax.set_ylabel("School")
        fig.tight_layout()
        st.pyplot(fig)

# --- Quality & indexing --------------------------------------------------
with tab_quality:
    c1, c2 = st.columns(2)
    with c1:
        q_counts = df_pubs["Quartile"].value_counts().reindex(quartiles, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(q_counts.index, q_counts.values, color=[PALETTE[3], PALETTE[1], PALETTE[4], PALETTE[6]])
        ax.set_title("SJR Quartile Distribution")
        ax.set_ylabel("Publications")
        for i, v in enumerate(q_counts.values):
            ax.text(i, v, str(int(v)), ha="center", va="bottom")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(f"{df_pubs['Quartile'].isna().sum()} publications have no reported SJR quartile (e.g. conference papers, books).")
    with c2:
        idx_exploded = explode_col(df_pubs, "Indexing List", "Indexing")
        idx_counts = idx_exploded["Indexing"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(idx_counts.index, idx_counts.values, color=PALETTE[5])
        ax.set_title("Indexing Status")
        ax.set_ylabel("Publications")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        st.pyplot(fig)

# --- Faculty / author contribution ---------------------------------------
with tab_people:
    top_n = st.slider("Show top N authors", 5, 30, 15)
    auth_counts = (
        df.explode("SaiU Author List")
        .dropna(subset=["SaiU Author List"])
        .groupby("SaiU Author List")["Pub Key"].nunique()
        .sort_values(ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(auth_counts))))
    ax.barh(auth_counts.index[::-1], auth_counts.values[::-1], color=PALETTE[0])
    ax.set_title(f"Top {top_n} Faculty by Publication Count")
    ax.set_xlabel("Publications")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("**🔎 Researcher profile**")
    if authors:
        profile_author = st.selectbox("Select a faculty member", ["—"] + authors)
        if profile_author != "—":
            adf = df[df["SaiU Author List"].apply(lambda l: profile_author in l)]
            adf_pubs = dedupe_publications(adf)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Publications", adf_pubs["Pub Key"].nunique())
            p2.metric("Scopus", int(adf_pubs["Is Scopus"].sum()))
            p3.metric("WoS", int(adf_pubs["Is WoS"].sum()))
            p4.metric("Q1 papers", int((adf_pubs["Quartile"] == "Q1").sum()))
            st.dataframe(
                adf_pubs[["Title", "Year", "Source title", "Document Type", "Quartile", "Best Link"]]
                .sort_values("Year", ascending=False),
                use_container_width=True, hide_index=True,
                column_config={"Best Link": st.column_config.LinkColumn("Link")},
            )

# --- SDG alignment ---------------------------------------------------------
with tab_sdg:
    sdg_exploded = explode_col(df_pubs, "SDG List", "SDG")
    if not sdg_exploded.empty:
        sdg_exploded["Label"] = sdg_exploded["SDG"].map(lambda n: f"SDG {n}\n{SDG_NAMES.get(n, '')}")
        sdg_counts = sdg_exploded.groupby("SDG")["SDG"].count()
        sdg_counts = sdg_counts.reindex(sorted(sdg_counts.index, key=int))
        labels = [f"SDG {n} — {SDG_NAMES.get(n, n)}" for n in sdg_counts.index]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(sdg_counts))))
        ax.barh(labels, sdg_counts.values, color=PALETTE[6])
        ax.set_title("Publications by SDG")
        ax.set_xlabel("Publications")
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("**SDG × School heatmap**")
        sdg_school = df_pubs.explode("SDG List").dropna(subset=["SDG List"])
        heat2 = sdg_school.groupby(["School", "SDG List"])["Pub Key"].nunique().unstack(fill_value=0)
        heat2 = heat2.reindex(sorted(heat2.columns, key=int), axis=1)
        if not heat2.empty:
            fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(heat2))))
            sns.heatmap(heat2, cmap="Greens", annot=True, fmt="d", cbar_kws={"label": "Publications"}, ax=ax)
            ax.set_xlabel("SDG"); ax.set_ylabel("School")
            fig.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No SDG data available for the current filter selection.")

st.markdown("---")

# --------------------------------------------------------------------------
# Publication Explorer
# --------------------------------------------------------------------------
st.subheader("🔍 Publication Explorer")

search = st.text_input("Search publications (title, author, keywords, publisher)")

explorer_df = df_pubs.copy()
if search:
    s = search.lower()
    explorer_df = explorer_df[
        explorer_df["Title"].str.lower().str.contains(s, na=False)
        | explorer_df["SaiU Authors"].str.lower().str.contains(s, na=False)
        | explorer_df.get("Keywords", pd.Series("", index=explorer_df.index)).fillna("").str.lower().str.contains(s, na=False)
        | explorer_df["Publisher"].str.lower().str.contains(s, na=False)
        | explorer_df["Source title"].str.lower().str.contains(s, na=False)
    ]

sort_col = st.selectbox(
    "Sort by", ["Year", "Title", "School", "Quartile", "Document Type"], index=0
)
sort_asc = st.checkbox("Ascending", value=False)
explorer_df = explorer_df.sort_values(sort_col, ascending=sort_asc, na_position="last")

display_cols = ["Title", "SaiU Authors", "School", "Year", "Document Type",
                 "Source title", "Quartile", "Indexing Status", "Best Link"]
display_cols = [c for c in display_cols if c in explorer_df.columns]

st.dataframe(
    explorer_df[display_cols],
    use_container_width=True,
    hide_index=True,
    height=450,
    column_config={"Best Link": st.column_config.LinkColumn("Link / DOI")},
)
st.caption(f"Showing {len(explorer_df):,} of {total_pubs:,} filtered publications.")

# Export
csv_buf = io.StringIO()
explorer_df[display_cols].to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Export current view as CSV",
    data=csv_buf.getvalue(),
    file_name="sai_university_publications_export.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "Built with Streamlit, Pandas, Matplotlib & Seaborn · "
    "Data source: Sai University Publications API"
)
