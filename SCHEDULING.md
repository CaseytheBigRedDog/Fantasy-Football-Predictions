# Scheduling the weekly refresh (Windows Task Scheduler)

`run_refresh.bat` re-downloads fresh data and reruns the full pipeline
(01 -> 04), logging everything to `refresh_log.txt` in the same folder so you
can check what happened without watching it run live.

## One-time setup

1. Make sure `refresh_data.py` and `run_refresh.bat` are in your
   Fantasy-Football-Predictions folder alongside everything else.
2. Open the Start menu, type **Task Scheduler**, open it.
3. Click **Create Basic Task** (right panel).
4. Name it something like "Fantasy Football Weekly Refresh" -> Next.
5. Trigger: choose **Weekly**, pick a day (Tuesday mornings are a good
   choice -- Monday Night Football is over, so the previous week's stats
   are final) -> Next.
6. Action: **Start a program** -> Next.
7. Program/script: click Browse, navigate to your
   Fantasy-Football-Predictions folder, select `run_refresh.bat` -> Next.
8. Finish.

## Checking it worked

After it's run (or to test it immediately -- right-click the task in Task
Scheduler and choose **Run**), open `refresh_log.txt` in the project folder.
You should see the same kind of output you saw running things manually:
row counts, MAE numbers, and the FantasyPros comparison table, all dated
at the top from `datetime.now()`.

## A few things worth knowing

- **In the off-season**, this will just re-pull the same data and retrain
  on the same 2013-2024(ish) games -- harmless, just not very useful. It
  becomes worth having scheduled once the season is underway.
- **Early in a season**, some files (like the current year's snap counts)
  may not exist yet -- the script detects this and prints a note rather
  than crashing, so a partial run won't break anything.
- **If your laptop is asleep/off** at the scheduled time, Task Scheduler
  by default just skips that run -- it won't run retroactively unless you
  check "Run task as soon as possible after a scheduled start is missed"
  under the task's Settings tab.
- **Log file grows over time** since new runs append rather than overwrite.
  Feel free to delete `refresh_log.txt` periodically if it gets long --
  it's not needed for the pipeline to work, just for your own visibility.
