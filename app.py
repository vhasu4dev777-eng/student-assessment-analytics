
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

st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.caption("Filter the analytics to explore student groups.")

# Course filter
# Human-readable labels for OULAD course modules
COURSE_NAMES = {
    "AAA": "Module AAA",
    "BBB": "Module BBB",
    "CCC": "Module CCC",
    "DDD": "Module DDD",
    "EEE": "Module EEE",
    "FFF": "Module FFF",
    "GGG": "Module GGG",
}

courses = sorted(df["code_module"].dropna().unique().tolist())



selected_courses = st.sidebar.multiselect(
    "Course module",
    options=courses,
    default=courses,
    help="Select one or more courses.",
)

# Presentation filter
presentations = sorted(
    df["code_presentation"].dropna().unique().tolist()
)

selected_presentations = st.sidebar.multiselect(
    "Course presentation",
    options=presentations,
    default=presentations,
    help="Select one or more course presentations.",
)

# Age filter
age_bands = sorted(df["age_band"].dropna().unique().tolist())

selected_age_bands = st.sidebar.multiselect(
    "Age band",
    options=age_bands,
    default=age_bands,
    help="Compare students across age groups.",
)

# Education filter
education_levels = sorted(
    df["highest_education"].dropna().unique().tolist()
)

selected_education = st.sidebar.multiselect(
    "Highest education",
    options=education_levels,
    default=education_levels,
    help="Compare students by education level.",
)

# Filter the dataset
filtered_df = df[
    df["code_module"].isin(selected_courses)
    & df["code_presentation"].isin(selected_presentations)
    & df["age_band"].isin(selected_age_bands)
    & df["highest_education"].isin(selected_education)
].copy()

# Reset filters button
if st.sidebar.button("Reset all filters"):
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    f"Showing {len(filtered_df):,} of {len(df):,} student records."
)

if filtered_df.empty:
    st.warning(
        "No student records match these filters. "
        "Select at least one option in each filter."
    )
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
    "code_module": "Course Module",
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


with st.expander("🔍 Explore the filtered student records", expanded=False):
    st.write(f"Showing {len(filtered_df):,} student records.")

    # Search for a student
    if "id_student" in filtered_df.columns:
        student_id = st.text_input(
            "Search by Student ID",
            placeholder="Enter a student ID..."
        )

        if student_id.strip():
            student_results = filtered_df[
                filtered_df["id_student"].astype(str).str.contains(
                    student_id.strip(),
                    case=False,
                    na=False
                )
            ]

            if not student_results.empty:
                st.success(f"Found {len(student_results)} matching record(s).")
                st.dataframe(student_results, width="stretch")
            else:
                st.warning("No matching students found.")
        else:
            st.dataframe(filtered_df, width="stretch")
    else:
        st.warning("Student ID column not found in the dataset.")

    # Download the filtered records
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download filtered records as CSV",
        data=csv,
        file_name="filtered_student_records.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# STUDENT RISK PREDICTION
# --------------------------------------------------

import joblib

MODEL_PATH = BASE_DIR / "models" / "student_risk_model.joblib"

st.divider()
st.header("🎯 Student Risk Prediction")

st.write(
    "Enter the student's academic and early engagement details "
    "to estimate their academic risk using the trained Random Forest model."
)

if not MODEL_PATH.exists():
    st.warning(
        "The trained model was not found. "
        "Run python src/save_trained_model.py first."
    )
else:
    @st.cache_resource
    def load_model(path):
        return joblib.load(path)

    model = load_model(str(MODEL_PATH))

    with st.form("student_risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            course = st.selectbox(
                "Course module",
                options=sorted(df["code_module"].dropna().unique()),
            )

            presentation = st.selectbox(
                "Course presentation",
                options=sorted(
                    df["code_presentation"].dropna().unique()
                ),
            )

            education = st.selectbox(
                "Highest education",
                options=sorted(
                    df["highest_education"].dropna().unique()
                ),
            )

            age_band = st.selectbox(
                "Age band",
                options=sorted(df["age_band"].dropna().unique()),
            )

        with col2:
            previous_attempts = st.number_input(
                "Number of previous attempts",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            studied_credits = st.number_input(
                "Studied credits",
                min_value=0,
                max_value=300,
                value=60,
                step=10,
            )

            early_clicks = st.number_input(
                "VLE clicks during first 30 days",
                min_value=0,
                max_value=100000,
                value=200,
                step=10,
            )

        submitted = st.form_submit_button(
            "Estimate Academic Risk",
            type="primary",
        )

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "clicks_first_30_days": early_clicks,
                    "num_of_prev_attempts": previous_attempts,
                    "studied_credits": studied_credits,
                    "age_band": age_band,
                    "highest_education": education,
                    "code_module": course,
                    "code_presentation": presentation,
                }
            ]
        )

        risk_probability = float(
            model.predict_proba(input_data)[0, 1]
        )

        threshold = 0.40
        predicted_at_risk = risk_probability >= threshold

        st.subheader("Prediction Result")

        metric_col1, metric_col2 = st.columns(2)

        metric_col1.metric(
            "Estimated Risk Probability",
            f"{risk_probability:.1%}",
        )

        metric_col2.metric(
            "Model Classification",
            "Higher estimated risk"
            if predicted_at_risk
            else "Lower estimated risk",
        )

        st.progress(risk_probability)

        if predicted_at_risk:
            st.warning(
                "The model has flagged this profile for possible follow-up. "
                "Consider reviewing the student's engagement and offering "
                "appropriate academic support."
            )
        else:
            st.success(
                "The model has not flagged this profile at the selected "
                "threshold. Continue monitoring student progress normally."
            )

        st.caption(
            "Experimental decision-support tool. The probability is a model "
            "estimate based on historical data, not a verified probability "
            "of future failure. Predictions may be inaccurate and should "
            "never be used alone to make decisions about students."
        )