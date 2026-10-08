-- Rolling average of each player's PRIOR 3 games (PPR fantasy points).
-- The window stops at "1 PRECEDING" so the current week is excluded.
-- That keeps the future out of the feature, which prevents data leakage.
WITH rolling AS (
    SELECT
        player_id,
        player_display_name,
        position,
        season,
        week,
        fantasy_points_ppr,
        AVG(fantasy_points_ppr) OVER (
            PARTITION BY player_id
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS avg_ppr_last_3
    FROM player_stats
)
SELECT *
FROM rolling
WHERE avg_ppr_last_3 IS NOT NULL
  AND season = 2026
ORDER BY season DESC, week DESC, avg_ppr_last_3 DESC;