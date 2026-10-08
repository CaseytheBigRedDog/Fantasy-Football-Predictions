-- Evaluates the model's point prediction ("expected") against the naive
-- baseline (average of the player's last 3 games), by week and position.
-- Predictions are joined to scored results on player + team + position.
-- MAE = mean absolute error. Positive pct_improvement = model beat baseline.
WITH joined AS (
    SELECT 3 AS week, s.position, p.expected, s.naive, s.actual
    FROM scored_2026_week3 AS s
    JOIN predictions_2026_week3 AS p
      ON p.player = s.player
     AND p.team = s.team
     AND p.position = s.position
    WHERE s.actual IS NOT NULL

    UNION ALL

    SELECT 4 AS week, s.position, p.expected, s.naive, s.actual
    FROM scored_2026_week4 AS s
    JOIN predictions_2026_week4 AS p
      ON p.player = s.player
     AND p.team = s.team
     AND p.position = s.position
    WHERE s.actual IS NOT NULL
)
SELECT
    week,
    position,
    COUNT(*) AS n_players,
    ROUND(AVG(ABS(expected - actual)), 2) AS model_mae,
    ROUND(AVG(ABS(naive - actual)), 2) AS naive_mae,
    ROUND(
        100.0 * (AVG(ABS(naive - actual)) - AVG(ABS(expected - actual)))
        / AVG(ABS(naive - actual)),
        1
    ) AS pct_improvement
FROM joined
GROUP BY week, position
ORDER BY week, position;