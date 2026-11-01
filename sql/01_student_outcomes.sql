-- Question: What is the distribution of student final results?

SELECT
    final_result,
    COUNT(*) AS student_count,
    ROUND(
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM studentInfo),
        2
    ) AS percentage
FROM studentInfo
GROUP BY final_result
ORDER BY student_count DESC;