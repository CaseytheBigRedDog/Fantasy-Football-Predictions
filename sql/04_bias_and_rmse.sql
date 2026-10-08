-- Pooled evaluation across weeks 3-4, by position.
-- bias = average (actual - prediction); positive means the model under-predicts.
-- RMSE penalizes large misses more heavily than MAE.
WITH joined AS (
    SELECT s.position, p.expected, p.median, s.naive, s.actual
    FROM scored_2026_week3 AS s
    JOIN predictions_2026_week3 AS p
      ON p.player = s.player AND p.team = s.team AND p.position = s.position
    WHERE s.actual IS NOT NULL

    UNION ALL

    SELECT s.position, p.expected, p.median, s.naive, s.actual
    FROM scored_2026_week4 AS s
    JOIN predictions_2026_week4 AS p
      ON p.player = s.player AND p.team = s.team AND p.position = s.position
    WHERE s.actual IS NOT NULL
)
SELECT
    position,
    COUNT(*) AS n,
    ROUND(AVG(actual - expected), 2) AS expected_bias,
    ROUND(AVG(actual - median), 2)   AS median_bias,
    ROUND(AVG(actual - naive), 2)    AS naive_bias,
    ROUND(SQRT(AVG((actual - expected) * (actual - expected))), 2) AS expected_rmse,
    ROUND(SQRT(AVG((actual - median) * (actual - median))), 2)     AS median_rmse,
    ROUND(SQRT(AVG((actual - naive) * (actual - naive))), 2)       AS naive_rmse
FROM joined
GROUP BY position
ORDER BY position;