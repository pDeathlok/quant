# Daily Refresh Monitor Recovery: 2026-09-08

## Result

- Entry point: `python3 scripts/run_daily_web_refresh.py`.
- Run: `all-20260908155626-5e22786c`.
- Started 15:56:26, finished 17:46:37 Asia/Shanghai; 6611.331 seconds.
- Success, 19/19 steps, failed_count=0, error_summary=null.
- Postflight: all 36 checked nodes passed, no freshness failures or pending nodes.
- Validated publication: `all-20260908155626-5e22786c-fad43708`.
- Manifest: `data/routine/20260908_155626_all-20260908155626-5e22786c/manifest.json`.

Market daily processed 5,558 symbols with zero failures through 20260908.
Daily_basic had missing vendor fields at 15:57 and 16:07; automatic ten-minute
retries recovered official complete data at 16:17:26, before the 17:20 deadline.
Live features, Chan, bond grid and allotments reached 2026-09-08.
Similar-pattern analysis covered 61 targets for 2026-09-08, generated 17:30:41.
The historical vector reference library was reused from 2026-09-07 23:04:48
under its explicit weekly/seven-day TTL contract; it was not rebuilt today.

## Monitor Failure and Repair

The initial sandbox invocation could not reach the local proxy and exited
before submission. The approved invocation started the single production run.
During selector processing, intermittent HTTP status read timeouts exhausted
the runner's ordinary failure counter and caused its process to exit. The
refresh owner remained alive and kept computing; no publication failure was
observed. The timeout was not caused by the Tushare 17:20 deadline.

The runner now retries transport errors within the existing no-progress budget,
without consuming business-task retry attempts or restarting the worker. It
does not reset that budget on transport failure. An explicit restart is also
suppressed if status cannot be read but a service still listens on the port.

After testing, the same entry point with `--no-restart-service` attached to the
original run and exited 0 after the owner released its publication. No lower
refresh stages were executed manually and no duplicate run was submitted.

## Verification

Python 3.13: 36 runner/lifecycle/transport tests passed.
Python 3.9: 32 runner/transport/CI contract tests passed.
Ruff fatal/undefined-name checks and `git diff --check` passed.
Six new transport tests were added to both CI versions. These repair changes
are local and have not been committed or pushed during this automation run.
