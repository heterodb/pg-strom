#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

PGVER="${PGVER:-}"
TEST_TIMEOUT_SECS="${TEST_TIMEOUT_SECS:-900}"
KNOWN_FAILURES_FILE="${KNOWN_FAILURES_FILE:-${SCRIPT_DIR}/known_failures.txt}"
TEST_COMMAND="${TEST_COMMAND:-}"
PSQLBIN_ROOT="${PSQLBIN_ROOT:-/actions-runner/psqlbin}"
PGDATA_BASE="${PGDATA_BASE:-/opt}"
LOG_BASE="${LOG_BASE:-/tmp}"

usage() {
  cat <<'USAGE'
Usage: run_ci_tests.sh --pgver <major.minor> [options]

Options:
  --pgver <ver>              PostgreSQL version suffix under pgbin-<ver> (e.g. 17.9)
  --timeout <secs>           Test timeout in seconds (default: 900)
  --known-failures <file>    File with known failing test names (one per line)
  --test-command <command>   Override test command
  --pgdata-base <dir>        Base dir for PGDATA (default: /opt)
  --log-base <dir>           Base dir for logs/socket dirs (default: /tmp)
  --help                     Show this help

Environment:
  PSQLBIN_ROOT               PostgreSQL install root (default: /actions-runner/psqlbin)
  PGVER                      Same as --pgver
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pgver)
      PGVER="${2:-}"
      shift 2
      ;;
    --timeout)
      TEST_TIMEOUT_SECS="${2:-}"
      shift 2
      ;;
    --known-failures)
      KNOWN_FAILURES_FILE="${2:-}"
      shift 2
      ;;
    --test-command)
      TEST_COMMAND="${2:-}"
      shift 2
      ;;
    --pgdata-base)
      PGDATA_BASE="${2:-}"
      shift 2
      ;;
    --log-base)
      LOG_BASE="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PGVER}" ]]; then
  echo "--pgver is required" >&2
  usage >&2
  exit 2
fi

PG_BINDIR="${PSQLBIN_ROOT}/pgbin-${PGVER}/bin"
PG_CONFIG="${PG_BINDIR}/pg_config"
INITDB="${PG_BINDIR}/initdb"
PG_CTL="${PG_BINDIR}/pg_ctl"
PSQL="${PG_BINDIR}/psql"
PG_ISREADY="${PG_BINDIR}/pg_isready"

for cmd in "${PG_CONFIG}" "${INITDB}" "${PG_CTL}" "${PSQL}" "${PG_ISREADY}" timeout make; do
  if ! command -v "${cmd}" >/dev/null 2>&1 && [[ ! -x "${cmd}" ]]; then
    echo "Required command not found: ${cmd}" >&2
    exit 1
  fi
done

RUN_TAG="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-${PGVER}-$$"
RUN_USER="$(id -un)"
PGDATA="${PGDATA_BASE}/github_actions_pgdata-${PGVER}-${RUN_USER}"
SOCKET_DIR="${LOG_BASE}/pgstrom-sock-${RUN_TAG}"
PG_LOG="${LOG_BASE}/pgstrom-postgres-${RUN_TAG}.log"
TEST_LOG="${LOG_BASE}/pgstrom-tests-${RUN_TAG}.log"
FAILED_LIST="${LOG_BASE}/pgstrom-failed-tests-${RUN_TAG}.txt"

ensure_dir_owned_by_me() {
  local dir="$1"
  if mkdir -p "${dir}" 2>/dev/null && [[ -w "${dir}" ]]; then
    return 0
  fi
  echo "Permission denied for ${dir}" >&2
  return 1
}

ensure_dir_owned_by_me "${SOCKET_DIR}"
ensure_dir_owned_by_me "${PGDATA}"

if [[ ! -w "${PGDATA}" ]]; then
  echo "PGDATA is not writable: ${PGDATA}" >&2
  exit 1
fi

POSTGRES_STARTED=0
SHOW_LOGS=0

cleanup() {
  local rc=$?
  if [[ ${POSTGRES_STARTED} -eq 1 ]]; then
    if "${PG_CTL}" -D "${PGDATA}" status >/dev/null 2>&1; then
      "${PG_CTL}" -D "${PGDATA}" -m fast -w stop || true
    fi
  fi
  rm -rf "${SOCKET_DIR}" || true

  if [[ ${SHOW_LOGS} -eq 1 || ${rc} -ne 0 ]]; then
    echo "===== PostgreSQL log: ${PG_LOG} ====="
    [[ -f "${PG_LOG}" ]] && cat "${PG_LOG}" || echo "(log file not found)"
    echo "===== Test log: ${TEST_LOG} ====="
    [[ -f "${TEST_LOG}" ]] && cat "${TEST_LOG}" || echo "(log file not found)"
  fi

  exit ${rc}
}
trap cleanup EXIT INT TERM

if [[ ! -f "${PGDATA}/PG_VERSION" ]]; then
  echo "Initializing PGDATA: ${PGDATA}"
  "${INITDB}" -D "${PGDATA}" >/dev/null
  cp "${SCRIPT_DIR}/postgresql.conf" "${PGDATA}/postgresql.conf"
else
  echo "Reusing existing PGDATA: ${PGDATA}"
fi

export PGHOST="${SOCKET_DIR}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-${RUN_USER}}"

START_OPTS="-c listen_addresses='' -c unix_socket_directories='${SOCKET_DIR}'"

echo "Starting PostgreSQL ${PGVER} with PGDATA=${PGDATA} PGHOST=${PGHOST}"
"${PG_CTL}" -D "${PGDATA}" -l "${PG_LOG}" -w start -o "${START_OPTS}"
POSTGRES_STARTED=1

if ! "${PG_ISREADY}" -h "${PGHOST}" -p "${PGPORT}" >/dev/null 2>&1; then
  SHOW_LOGS=1
  echo "PostgreSQL is not ready" >&2
  exit 1
fi

if [[ -n "${TEST_COMMAND}" ]]; then
  RUN_TEST_CMD=(bash -lc "${TEST_COMMAND}")
else
  RUN_TEST_CMD=(make -C "${REPO_ROOT}/test" PG_CONFIG="${PG_CONFIG}" installcheck)
fi

echo "Running tests with timeout=${TEST_TIMEOUT_SECS}s (attempt 1)"
set +e
timeout --signal=TERM --kill-after=30s "${TEST_TIMEOUT_SECS}" "${RUN_TEST_CMD[@]}" 2>&1 | tee "${TEST_LOG}"
TEST_RC=${PIPESTATUS[0]}
set -e

if [[ ${TEST_RC} -ne 0 && ${TEST_RC} -ne 124 ]]; then
  echo "Initial test run failed (exit=${TEST_RC}); reinitializing pg_strom extension and retrying once"
  if ! "${PSQL}" -v ON_ERROR_STOP=1 -d postgres -c "create extension if not exists pg_strom;" >/dev/null; then
    SHOW_LOGS=1
    echo "Failed to prepare pg_strom extension for retry" >&2
    exit 1
  fi
  if ! "${PSQL}" -v ON_ERROR_STOP=1 -d postgres -c "drop extension pg_strom cascade; create extension pg_strom;" >/dev/null; then
    SHOW_LOGS=1
    echo "Failed to reinitialize pg_strom extension for retry" >&2
    exit 1
  fi

  {
    echo
    echo "===== Retry after pg_strom reinitialization ====="
  } >> "${TEST_LOG}"

  echo "Running tests with timeout=${TEST_TIMEOUT_SECS}s (attempt 2)"
  set +e
  timeout --signal=TERM --kill-after=30s "${TEST_TIMEOUT_SECS}" "${RUN_TEST_CMD[@]}" 2>&1 | tee -a "${TEST_LOG}"
  TEST_RC=${PIPESTATUS[0]}
  set -e
fi

if [[ ${TEST_RC} -eq 124 ]]; then
  SHOW_LOGS=1
  echo "Test execution timed out after ${TEST_TIMEOUT_SECS} seconds" >&2
  exit 1
fi

if ! "${PG_CTL}" -D "${PGDATA}" status >/dev/null 2>&1; then
  SHOW_LOGS=1
  echo "PostgreSQL exited unexpectedly during tests" >&2
  exit 1
fi

if [[ ${TEST_RC} -eq 0 ]]; then
  echo "Tests passed"
  exit 0
fi

# Parse failed test names from pg_regress summary and per-test output.
: > "${FAILED_LIST}"
grep -E '^not ok[[:space:]]+[0-9]+[[:space:]]+-[[:space:]]+' "${TEST_LOG}" \
  | sed -E 's/^not ok[[:space:]]+[0-9]+[[:space:]]+-[[:space:]]+//' \
  | awk '{print $1}' >> "${FAILED_LIST}" || true
grep -E '^test[[:space:]]+[^[:space:]]+[[:space:]]+\.\.\.[[:space:]]+FAILED$' "${TEST_LOG}" | awk '{print $2}' >> "${FAILED_LIST}" || true
grep -E '^FAILED test\(s\):' "${TEST_LOG}" | sed -E 's/^FAILED test\(s\):[[:space:]]*//' | tr ' ' '\n' >> "${FAILED_LIST}" || true
sort -u -o "${FAILED_LIST}" "${FAILED_LIST}"

if [[ ! -s "${FAILED_LIST}" ]]; then
  SHOW_LOGS=1
  echo "Tests failed but failed test names could not be parsed" >&2
  exit 1
fi

if [[ ! -f "${KNOWN_FAILURES_FILE}" ]]; then
  SHOW_LOGS=1
  echo "Known failures file not found: ${KNOWN_FAILURES_FILE}" >&2
  exit 1
fi

UNKNOWN_FAILURES=$(grep -vxF -f "${KNOWN_FAILURES_FILE}" "${FAILED_LIST}" || true)
if [[ -n "${UNKNOWN_FAILURES}" ]]; then
  SHOW_LOGS=1
  echo "Unexpected test failures detected:" >&2
  echo "${UNKNOWN_FAILURES}" >&2
  exit 1
fi

echo "Only known failures were detected; treating as success"
exit 0
