-- Breaks down the scored tables by their own status label.
-- Shows how many rows have no actual score, how many are exactly 0,
-- and the average actual score in each group.
WITH scored AS (
    SELECT 3 AS week, player, position, status AS result_status, actual
    FROM scored_2026_week3
    UNION ALL
    SELECT 4 AS week, player, position, status AS result_status, actual
    FROM scored_2026_week4
)
SELECT
    result_status,
    COUNT(*) AS n,
    SUM(CASE WHEN actual IS NULL THEN 1 ELSE 0 END) AS n_missing_actual,
    SUM(CASE WHEN actual = 0 THEN 1 ELSE 0 END) AS n_zero,
    ROUND(AVG(actual), 2) AS avg_actual
FROM scored
GROUP BY result_status
ORDER BY n DESC;