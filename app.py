
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Assessment Analytics",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "model_dataset.csv"

st.title("🎓 Student Assessment Analytics")
st.markdown(
    "Explore student performance, early engagement, "
    "and academic risk using the Open University Learning Analytics Dataset."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Ensure expected numeric columns are numeric.
    for col in [
        "clicks_first_30_days",
        "at_risk",
        "studied_credits",
        "num_of_prev_attempts",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Make sure model_dataset.csv exists inside "
        "data/processed/."
    )
    st.stop()

df = load_data(DATA_PATH)

required_columns = {
    "at_risk",
    "clicks_first_30_days",
    "code_module",
    "final_result",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(f"Missing required columns: {sorted(missing_columns)}")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

courses = sorted(df["code_module"].dropna().unique().tolist())

selected_courses = st.sidebar.multiselect(
    "Select course(s)",
    options=courses,
    default=courses,
)

filtered_df = df[df["code_module"].isin(selected_courses)].copy()

if filtered_df.empty:
    st.warning("No student records match the selected filters.")
    st.stop()

# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------

total_students = len(filtered_df)
at_risk_students = int(filtered_df["at_risk"].eq(1).sum())
at_risk_rate = (
    at_risk_students / total_students * 100
    if total_students
    else 0
)

average_clicks = filtered_df["clicks_first_30_days"].mean()

st.subheader("📊 Student Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Student Records", f"{total_students:,}")
col2.metric("At-Risk Records", f"{at_risk_students:,}")
col3.metric("At-Risk Rate", f"{at_risk_rate:.1f}%")
col4.metric(
    "Average Early Clicks",
    f"{average_clicks:,.1f}" if pd.notna(average_clicks) else "N/A",
)

st.caption(
    "Risk labels are derived from final outcomes: Fail or Withdrawn = at risk. "
    "These are retrospective labels, not verified predictions of future failure."
)

st.divider()

# --------------------------------------------------
# ACADEMIC OUTCOMES
# --------------------------------------------------

st.subheader("📚 Academic Performance")

left, right = st.columns(2)

with left:
    st.markdown("**Distribution of Final Outcomes**")

    outcome_counts = (
        filtered_df["final_result"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Final Outcome")
        .reset_index(name="Students")
    )

    fig_outcomes = px.pie(
        outcome_counts,
        names="Final Outcome",
        values="Students",
        hole=0.45,
        title="Final Outcome Distribution",
    )

    fig_outcomes.update_layout(legend_title_text="Outcome")
    st.plotly_chart(fig_outcomes, use_container_width=True)

with right:
    st.markdown("**At-Risk Rate by Course**")

    course_risk = (
        filtered_df.groupby("code_module", observed=True)
        .agg(
            total_students=("at_risk", "size"),
            at_risk_students=("at_risk", "sum"),
        )
        .reset_index()
    )

    course_risk["at_risk_rate"] = (
        course_risk["at_risk_students"]
        / course_risk["total_students"]
        * 100
    )

    fig_course = px.bar(
        course_risk,
        x="code_module",
        y="at_risk_rate",
        text=course_risk["at_risk_rate"].map(lambda x: f"{x:.1f}%"),
        labels={
            "code_module": "Course",
            "at_risk_rate": "At-Risk Rate (%)",
        },
        title="At-Risk Rate by Course",
    )

    fig_course.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig_course, use_container_width=True)

st.divider()

# --------------------------------------------------
# EARLY ENGAGEMENT
# --------------------------------------------------

st.subheader("📈 Early Student Engagement")

engagement_df = filtered_df.copy()

engagement_df["engagement_group"] = pd.cut(
    engagement_df["clicks_first_30_days"],
    bins=[-1, 0, 100, 300, 1000, float("inf")],
    labels=[
        "No clicks",
        "1–100",
        "101–300",
        "301–1,000",
        "1,000+",
    ],
)

engagement_summary = (
    engagement_df.groupby("engagement_group", observed=False)
    .agg(
        total_students=("at_risk", "size"),
        at_risk_students=("at_risk", "sum"),
    )
    .reset_index()
)

engagement_summary["at_risk_rate"] = (
    engagement_summary["at_risk_students"]
    / engagement_summary["total_students"].replace(0, pd.NA)
    * 100
)

engagement_summary["engagement_group"] = (
    engagement_summary["engagement_group"].astype(str)
)

fig_engagement = px.bar(
    engagement_summary,
    x="engagement_group",
    y="at_risk_rate",
    text=engagement_summary["at_risk_rate"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    ),
    hover_data=["total_students", "at_risk_students"],
    labels={
        "engagement_group": "VLE Clicks in First 30 Days",
        "at_risk_rate": "At-Risk Rate (%)",
        "total_students": "Total Student Records",
        "at_risk_students": "At-Risk Records",
    },
    title="At-Risk Rate by Early Engagement",
)

fig_engagement.update_layout(yaxis_range=[0, 100])
st.plotly_chart(fig_engagement, use_container_width=True)

st.caption(
    "The chart shows an association between early engagement and final outcomes. "
    "It does not establish that low engagement causes an adverse outcome."
)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("🔍 Explore the filtered student records"):
    st.write(f"Showing {len(filtered_df):,} student records.")
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="student_assessment_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Student Assessment Analytics | Built with Python, Streamlit and Plotly")
