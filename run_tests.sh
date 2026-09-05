#!/usr/bin/env bash
# Every suite, one command. Run before any deploy: a solver bug that reaches
# the server costs rating, and five invalid moves forfeit a game outright.
set -uo pipefail
cd "$(dirname "$0")"
rc=0
for suite in tests/test_solvers.py tests/test_analysis.py tests/test_messages.py tests/test_advisor.py; do
  echo "=== $suite"
  python3 "$suite" || rc=1
done
exit $rc
