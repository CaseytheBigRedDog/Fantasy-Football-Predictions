-- Which predicted players have no stat line, by depth-chart slot?
-- Helps separate "injured/inactive" from "backup who never got snaps".
-- Also shows the range of p_play within each slot.
WITH joined AS (
    SELECT p.depth, p.p_play,
           CASE WHEN s.actual IS NOT NULL THEN 1 ELSE 0 END AS played
    FROM predictions_2026_week3 AS p
    JOIN scored_2026_week3 AS s
      ON s.player = p.player AND s.team = p.team AND s.position = p.position

    UNION ALL

    SELECT p.depth, p.p_play,
           CASE WHEN s.actual IS NOT NULL THEN 1 ELSE 0 END AS played
    FROM predictions_2026_week4 AS p
    JOIN scored_2026_week4 AS s
      ON s.player = p.player AND s.team = p.team AND s.position = p.position
)
SELECT
    COALESCE(CAST(depth AS TEXT), 'unknown') AS depth_slot,
    COUNT(*) AS n,
    ROUND(AVG(played), 2) AS play_rate,
    MIN(p_play) AS min_p_play,
    MAX(p_play) AS max_p_play
FROM joined
GROUP BY COALESCE(CAST(depth AS TEXT), 'unknown')
ORDER BY n DESC;