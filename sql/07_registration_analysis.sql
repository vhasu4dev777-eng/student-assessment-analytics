-- Analyse registration timing and withdrawal outcomes.

SELECT
    final_result,
    COUNT(*) AS total_students,

    ROUND(
        AVG(date_registration),
        2
    ) AS average_registration_date,

    SUM(
        CASE
            WHEN date_unregistration IS NOT NULL THEN 1
            ELSE 0
        END
    ) AS students_with_unregistration_date,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN date_unregistration IS NOT NULL THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS unregistration_percentage

FROM studentRegistration
JOIN studentInfo
    ON studentRegistration.id_student = studentInfo.id_student
    AND studentRegistration.code_module = studentInfo.code_module
    AND studentRegistration.code_presentation = studentInfo.code_presentation

GROUP BY final_result
ORDER BY unregistration_percentage DESC;