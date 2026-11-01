-- Compare assessment performance across final-result categories.

SELECT
    si.final_result,
    COUNT(DISTINCT si.id_student || '-' ||
        si.code_module || '-' ||
        si.code_presentation) AS student_count,

    COUNT(sa.id_assessment) AS recorded_assessment_scores,

    ROUND(AVG(sa.score), 2) AS average_score,

    ROUND(MIN(sa.score), 2) AS minimum_score,

    ROUND(MAX(sa.score), 2) AS maximum_score

FROM studentInfo AS si

LEFT JOIN assessments AS a
    ON si.code_module = a.code_module
    AND si.code_presentation = a.code_presentation

LEFT JOIN studentAssessment AS sa
    ON sa.id_assessment = a.id_assessment
    AND sa.id_student = si.id_student

GROUP BY si.final_result
ORDER BY average_score DESC;