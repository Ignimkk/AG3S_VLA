# Historical T43T test harness

Source: original untracked `outputs/impl/T43T/p` captured in the migration backup, plus its T43R2 replay dependency.
These helper files retain the original numerical replay logic. Imports use this
checkout's canonical AG3S package, model assets use its robots directory, and
immutable raw recording inputs use `AG3S_LEGACY_STORAGE`. This is a test harness,
not a Panda benchmark runner. The saved off bit-identity mismatch was already
reproduced in the original workspace and is not hidden by changing the method.
