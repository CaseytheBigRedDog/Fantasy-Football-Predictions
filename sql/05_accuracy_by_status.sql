-- Model accuracy by injury designation (status from the predictions table).
-- bias = average (actual - expected); positive means the model under-predicts.
WITH joined AS (
    SELECT p.status, p.expected, s.naive, s.actual
    FROM scored_2026_week3 AS s
    JOIN predictions_2026_week3 AS p
      ON p.player = s.player AND p.team = s.team AND p.position = s.position
    WHERE s.actual IS NOT NULL

    UNION ALL

    SELECT p.status, p.expected, s.naive, s.actual
    FROM scored_2026_week4 AS s
    JOIN predictions_2026_week4 AS p
      ON p.player = s.player AND p.team = s.team AND p.position = s.position
    WHERE s.actual IS NOT NULL
)
SELECT
    COALESCE(status, 'none') AS injury_status,
    COUNT(*) AS n,
    ROUND(AVG(ABS(expected - actual)), 2) AS model_mae,
    ROUND(AVG(ABS(naive - actual)), 2)    AS naive_mae,
    ROUND(AVG(actual - expected), 2)      AS bias
FROM joined
GROUP BY COALESCE(status, 'none')
ORDER BY n DESC;