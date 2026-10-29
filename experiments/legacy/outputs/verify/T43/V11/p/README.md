# Legacy RB-Y1 V11 launchers

These are the original V11A/V11AB/V11ABC evaluation launchers and episode lists,
ported to this checkout's paths. They do not implement Panda benchmark rollouts.
The method code is the current branch, whose source version is in the migration
manifest; the original historical V11 snapshot is documented in legacy evidence.

`source env.sh` selects this checkout, separate outputs and unchanged legacy
checkpoint inputs. Set `AG3S_OPENPI_PYTHON` to a separately prepared GPU/VLA
interpreter. No GPU environment or model inference was launched during migration.

Online outputs now use `$AG3S_STORAGE_ROOT/results/benchmark/<KST timestamp_commit_run-id>/`.
Source `env.sh` once, preserve/export the printed `AG3S_RUN_DIR` for all related commands,
and start a new run after changing commits. Manifest executions capture actual server/client
arguments and checkpoint paths. Input obstacle configs remain in shared storage; input
recordings are never output destinations. See the live environment report for installation.

Archived replay/analysis scripts still require missing historical helpers and old output
layouts. They are not a completed postprocessing pipeline for new runs. Restore their
frozen helpers into this checkout and port input/output addresses before using them.
