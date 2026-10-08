-- Calibration of p_play: do players given an X% chance of playing
-- actually appear in the stats about X% of the time?
-- "played" = the player has a stat line (actual is not NULL).
WITH joined AS (
    SELECT p.p_play,
           CASE WHEN s.actual IS NOT NULL THEN 1 ELSE 0 END AS played
    FROM predictions_2026_week3 AS p
    JOIN scored_2026_week3 AS s
      ON s.player = p.player AND s.team = p.team AND s.position = p.position

    UNION ALL

    SELECT p.p_play,
           CASE WHEN s.actual IS NOT NULL THEN 1 ELSE 0 END AS played
    FROM predictions_2026_week4 AS p
    JOIN scored_2026_week4 AS s
      ON s.player = p.player AND s.team = p.team AND s.position = p.position
)
SELECT
    CASE
        WHEN p_play < 0.25 THEN '1: under 25%'
        WHEN p_play < 0.50 THEN '2: 25-50%'
        WHEN p_play < 0.75 THEN '3: 50-75%'
        WHEN p_play < 0.90 THEN '4: 75-90%'
        ELSE '5: 90%+'
    END AS p_play_bin,
    COUNT(*) AS n,
    ROUND(AVG(p_play), 2) AS avg_predicted_prob,
    ROUND(AVG(played), 2) AS actual_play_rate
FROM joined
GROUP BY 1
ORDER BY 1;