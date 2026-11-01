-- Analyse student outcomes by age group.

SELECT
    age_band,
    COUNT(*) AS total_students,

    SUM(
        CASE
            WHEN final_result = 'Pass' THEN 1
            ELSE 0
        END
    ) AS passed,

    SUM(
        CASE
            WHEN final_result = 'Distinction' THEN 1
            ELSE 0
        END
    ) AS distinction,

    SUM(
        CASE
            WHEN final_result = 'Fail' THEN 1
            ELSE 0
        END
    ) AS failed,

    SUM(
        CASE
            WHEN final_result = 'Withdrawn' THEN 1
            ELSE 0
        END
    ) AS withdrawn,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN final_result IN ('Fail', 'Withdrawn') THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS at_risk_percentage

FROM studentInfo

GROUP BY age_band

ORDER BY at_risk_percentage DESC;