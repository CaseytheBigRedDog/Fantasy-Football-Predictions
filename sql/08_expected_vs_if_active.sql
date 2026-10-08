-- For players who played, compares the model's two point predictions.
-- expected = (presumably) discounted by the chance of not playing.
-- expected_if_active = prediction assuming the player plays.
-- bias = average (actual - prediction); positive means under-prediction.
WITH joined AS (
    SELECT s.position, p.expected, p.expected_if_active, s.naive, s.actual
    FROM scored_2026_week3 AS s
    JOIN predictions_2026_week3 AS p
      ON p.player = s.player AND p.team = s.team AND p.position = s.position
    WHERE s.actual IS NOT NULL

    UNION ALL

    SELECT s.position, p.expected, p.expected_if_active, s.naive, s.actual
    FROM scored_2026_week4 AS s
    JOIN predictions_2026_week4 AS p
      ON p.player = s.player AND p.team = s.team AND p.position = s.position
    WHERE s.actual IS NOT NULL
)
SELECT
    position,
    COUNT(*) AS n,
    ROUND(AVG(ABS(expected - actual)), 2)           AS expected_mae,
    ROUND(AVG(ABS(expected_if_active - actual)), 2) AS if_active_mae,
    ROUND(AVG(ABS(naive - actual)), 2)              AS naive_mae,
    ROUND(AVG(actual - expected), 2)                AS expected_bias,
    ROUND(AVG(actual - expected_if_active), 2)      AS if_active_bias
FROM joined
GROUP BY position
ORDER BY position;