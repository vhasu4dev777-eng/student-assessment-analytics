-- Question: How do assessment results vary across students?

SELECT
    si.id_student,
    si.code_module,
    si.code_presentation,
    si.final_result,

    COUNT(sa.id_assessment) AS assessments_submitted,
    ROUND(AVG(sa.score), 2) AS average_score

FROM studentInfo AS si

LEFT JOIN assessments AS a
    ON si.code_module = a.code_module
    AND si.code_presentation = a.code_presentation

LEFT JOIN studentAssessment AS sa
    ON sa.id_assessment = a.id_assessment
    AND sa.id_student = si.id_student

GROUP BY
    si.id_student,
    si.code_module,
    si.code_presentation,
    si.final_result

ORDER BY average_score DESC
LIMIT 20;