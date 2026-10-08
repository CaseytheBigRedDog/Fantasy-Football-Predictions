-- Model vs naive accuracy by depth-chart tier, for players who played.
-- Slot 1 = starters (WR1, RB1, QB1, TE1); Slot 2/3 = next men up; Slot 4+ = deep depth.
-- bias = average (actual - expected); positive means under-prediction.
WITH joined AS (
    SELECT p.depth, p.expected, s.naive, s.actual
    FROM scored_2026_week3 AS s
    JOIN predictions_2026_week3 AS p
      ON p.player = s.player AND p.team = s.team AND p.position = s.position
    WHERE s.actual IS NOT NULL

    UNION ALL

    SELECT p.depth, p.expected, s.naive, s.actual
    FROM scored_2026_week4 AS s
    JOIN predictions_2026_week4 AS p
      ON p.player = s.player AND p.team = s.team AND p.position = s.position
    WHERE s.actual IS NOT NULL
),
tiered AS (
    SELECT
        CASE
            WHEN depth IS NULL THEN 'unknown'
            WHEN depth LIKE '%1' THEN 'Slot 1'
            WHEN depth LIKE '%2' THEN 'Slot 2'
            WHEN depth LIKE '%3' THEN 'Slot 3'
            ELSE 'Slot 4+'
        END AS depth_tier,
        expected, naive, actual
    FROM joined
)
SELECT
    depth_tier,
    COUNT(*) AS n,
    ROUND(AVG(actual), 2) AS avg_actual,
    ROUND(AVG(ABS(expected - actual)), 2) AS model_mae,
    ROUND(AVG(ABS(naive - actual)), 2)    AS naive_mae,
    ROUND(AVG(actual - expected), 2)      AS bias
FROM tiered
GROUP BY depth_tier
ORDER BY depth_tier;