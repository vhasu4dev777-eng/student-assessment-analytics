# Student Assessment Analytics and Early At-Risk Prediction

## Project Overview

This project uses SQL, Python, and machine learning to analyze student academic outcomes and investigate whether students at risk of failing or withdrawing can be identified using early learning engagement and student information.

The goal is to develop an early-warning prototype that could help educational institutions identify students who may benefit from additional academic support.

## Problem Statement

Student withdrawal and academic failure can be difficult to address if potential problems are identified too late.

This project analyzes historical student data to:
- Identify patterns in student academic outcomes.
- Explore relationships between early online learning engagement and final results.
- Build a binary classification model to identify at-risk student records.
- Evaluate the model using metrics relevant to an early-warning system.

## Dataset

**Dataset:** Open University Learning Analytics Dataset (OULAD)

The dataset contains student demographic information, course presentations, assessment results, registration information, virtual learning environment interactions, and final student outcomes.

The project uses the following data sources:
- `studentInfo.csv`
- `studentAssessment.csv`
- `studentRegistration.csv`
- `assessments.csv`
- `courses.csv`
- `vle.csv`
- `studentVle.csv`

The large `studentVle.csv` file is processed in chunks to reduce memory usage.

## Technology Stack

- **Python** — data processing and machine learning
- **SQL / SQLite** — relational data storage and analysis
- **Pandas and NumPy** — data manipulation
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Matplotlib and Seaborn** — data visualization
- **Git** — version control

## Project Workflow

1. Download and inspect the OULAD dataset.
2. Load six CSV files into a local SQLite database.
3. Perform SQL analyses of student outcomes, course performance, education level, previous attempts, registration information, and age groups.
4. Process virtual learning environment interactions in chunks.
5. Aggregate clicks during the first 30 days of a course presentation.
6. Join engagement data with student information.
7. Define the target variable:
   - `1` — final result is Fail or Withdrawn.
   - `0` — final result is Pass or Distinction.
8. Train and evaluate baseline, Logistic Regression, and Random Forest models.
9. Analyze feature importance and compare classification thresholds.
10. Evaluate the Random Forest using a student-level train/test split.

## Feature Engineering

The primary engagement feature is:

`clicks_first_30_days`

It measures the total recorded virtual learning environment clicks during days 0 through 30 of a course presentation.

Additional model features include:
- Number of previous attempts
- Studied credits
- Age band
- Highest education level
- Course module
- Course presentation

Categorical variables are encoded using `OneHotEncoder`.

The target variable is derived from `final_result`. That column is excluded from the model features to avoid target leakage.

## Model Development

Three approaches were evaluated during development:

- Dummy classifier baseline
- Logistic Regression using early engagement
- Random Forest using early engagement alone and then multiple features

The multi-feature Random Forest achieved the following results using a random train/test split:

| Metric | Result |
|---|---:|
| Accuracy | 72.47% |
| F1 score | 72.51% |
| ROC-AUC | 79.83% |

These results are preliminary and should not be treated as the final estimate of performance on unseen students.

## Student-Level Evaluation

A `GroupShuffleSplit` was used to keep each student ID entirely within either the training set or the test set.

At the provisional probability threshold of 0.40, the model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 67.07% |
| Precision | 63.74% |
| Recall | 86.49% |
| F1 score | 73.39% |
| ROC-AUC | 79.07% |

Confusion matrix:

| Actual class | Predicted not at risk | Predicted at risk |
|---|---:|---:|
| Not at risk | 1,401 | 1,672 |
| At risk | 459 | 2,939 |

The lower threshold increases recall, identifying more at-risk records, but also increases false positives.

The 0.40 threshold is a provisional choice for this project, not a universally optimal threshold. It should be selected according to the intended use and the relative costs of missed cases and false alarms.

## Feature Importance

The current Random Forest's impurity-based feature importance scores were:

| Feature | Importance |
|---|---:|
| Early engagement clicks | 65.34% |
| Course module | 11.27% |
| Highest education | 10.07% |
| Studied credits | 6.83% |
| Previous attempts | 3.24% |
| Course presentation | 1.94% |
| Age band | 1.32% |

Early engagement was the strongest feature in this model. These scores describe how the fitted model uses the features; they do not establish causation.

## Limitations

- Fail and Withdrawn are grouped into one at-risk category, although withdrawal can occur for reasons unrelated to academic performance.
- Recorded clicks are an imperfect measure of engagement and may not capture all learning activity.
- The model uses historical outcomes to define the target.
- The student-level split tests generalization to unseen student IDs, but does not fully test performance on future course presentations.
- The classification threshold has not yet been validated against real intervention costs.
- The feature importance scores are model-dependent and should not be interpreted as causal effects.
- The model is a prototype and has not been validated for real-world student interventions.

## Future Improvements

- Perform time-based evaluation on later course presentations.
- Compare additional models and tune hyperparameters using training data only.
- Evaluate precision-recall trade-offs at different thresholds.
- Explore calibration of predicted probabilities.
- Create visualizations of SQL analysis and model performance.
- Build a dashboard to explore student engagement and risk predictions.
- Improve the documentation and add reproducible setup instructions.

## Reproducibility

The Python scripts are located in the `src/` directory, and SQL queries are located in the `sql/` directory.

The raw dataset and generated database are kept out of version control. Download the OULAD dataset separately and place its CSV files in `data/raw/oulad/` before running the data-loading scripts.

Refer to the individual scripts for their execution order and expected inputs.

## Responsible Use

Predictions should be used only as decision-support signals. They should not be treated as definitive judgments about a student's ability or future, and they should not be used to deny educational opportunities. Any real-world intervention should involve appropriate human review and additional validation.