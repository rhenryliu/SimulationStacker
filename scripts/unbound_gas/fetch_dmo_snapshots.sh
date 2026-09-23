#! /bin/bash -l

# Download the dark-matter-only (DMO) reference snapshots for the matter power
# spectrum section of the unbound gas paper, all at z ~ 0.5:
#
#   SIMBA m100n1024 dm, snapshot 009 (z=0.4904, matches hydro snapshot 125):
#       1 file, 38.7 GB, public http, no key.
#   Illustris-1-Dark, snapshot 103 (z=0.503):
#       128 files, 217 GB, TNG API (key required).
#   TNG300-1-Dark, snapshot 67 (z=0.503):
#       75 files, 1.0 TB in total. An August download left 45 files truncated
#       and file 4 missing; only files that are missing, the wrong size or
#       unreadable are fetched (~610 GB).
#
# Each file is downloaded into a sibling "<dir>_staging/" directory (resumable
# with curl -C -), size-checked against the server's Content-Length, verified
# with verify_snapshot_files.py (opens, row counts, last-row reads), and only
# then moved onto its final path with an atomic rename. A truncated file already
# in place is therefore replaced only by a verified complete copy, never by a
# partial one. Finally each snapshot is checked for completeness (file count and
# particle totals against the header).
#
# The TNG API key is read from a header file (default
# ~/.config/tng/api_header, containing "api-key: <key>") via curl -H @file, so
# the key never appears on a command line, in `ps`, or in these logs.
#
# xfer QOS: free, runs on login nodes, meant for data staging (48 h cap). It
# must be paired with the "cron" architecture (-C cron); sbatch rejects -C cpu
# or no constraint. `sbatch --test-only` quotes a start a month out, but that
# estimate is a placeholder: xfer jobs pending in the queue wait only on
# dependencies, not resources.
# Safe to resubmit after a timeout: finished files are skipped and partial
# ones resume.
#
# Submit from the scripts/ directory:
#   sbatch unbound_gas/fetch_dmo_snapshots.sh
#   DATASETS="simba illustris" sbatch unbound_gas/fetch_dmo_snapshots.sh

#SBATCH -A desi
#SBATCH --qos=xfer
#SBATCH -C cron
#SBATCH --time=24:00:00
#SBATCH --job-name=fetch_dmo
#SBATCH -o ../Outputs_Perlmutter/fetch_dmo-%j.out

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

# The cosmodesi environment prepends a conda libcurl to LD_LIBRARY_PATH that is
# older than the system curl expects ("undefined symbol: curl_easy_ssls_export"),
# so curl runs with the pre-activation library path via sys_curl below; only
# the Python verification steps need the environment.
export SYS_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

sys_curl () {
    LD_LIBRARY_PATH="$SYS_LD_LIBRARY_PATH" /usr/bin/curl "$@"
}
export -f sys_curl

DATA_ROOT=${SIMSTACK_DATA_ROOT:-/pscratch/sd/r/rhliu/simulations/}
DATA_ROOT=${DATA_ROOT%/}
export TNG_API_HEADER=${TNG_API_HEADER:-$HOME/.config/tng/api_header}
DATASETS=${DATASETS:-"simba illustris tng300"}
NPAR=${NPAR:-4}   # concurrent downloads

TNG_API=https://www.tng-project.org/api
SIMBA_URL=http://simba.roe.ac.uk/simdata/m100n1024/dm/snapshots/snap_m100n1024_009.hdf5

if [[ " $DATASETS " =~ " illustris " || " $DATASETS " =~ " tng300 " ]]; then
    if [ ! -s "$TNG_API_HEADER" ]; then
        echo "TNG API header file $TNG_API_HEADER missing or empty" >&2
        exit 1
    fi
fi

# fetch_one URL DEST AUTH  (AUTH = tng | none)
fetch_one () {
    local url=$1 dest=$2 auth=$3
    local name stage part remote size attempt
    local -a hdr=()
    name=$(basename "$dest")
    stage="$(dirname "$dest")_staging"
    part="$stage/$name.part"
    [ "$auth" = "tng" ] && hdr=(-H "@$TNG_API_HEADER")
    mkdir -p "$stage"

    # Final-hop Content-Length (the TNG API answers with a 302 to a data server):
    # reset at every status line so a length from an earlier hop never survives.
    remote=$(sys_curl -sIL -m 120 "${hdr[@]}" "$url" | tr -d '\r' \
             | awk '/^HTTP\// {n=""} tolower($1)=="content-length:" {n=$2} END {print n}')
    if ! [[ "$remote" =~ ^[0-9]+$ ]] || [ "$remote" -eq 0 ]; then
        echo "FAIL $name: no Content-Length from server"
        return 1
    fi

    if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$remote" ] \
       && python unbound_gas/verify_snapshot_files.py check "$dest" > /dev/null 2>&1; then
        echo "OK   $name: already present and verified"
        return 0
    fi

    # A partial larger than the file (stale, or a server that ignored the Range
    # request and appended a full body) can never shrink back: start over.
    size=$(stat -c %s "$part" 2>/dev/null || echo 0)
    if [ "$size" -gt "$remote" ]; then
        echo "     $name: oversized partial ($size > $remote bytes), restarting"
        rm -f "$part"
        size=0
    fi

    attempt=0
    while [ "$size" -lt "$remote" ] && [ "$attempt" -lt 10 ]; do
        attempt=$((attempt + 1))
        sys_curl -sSL "${hdr[@]}" -C - --connect-timeout 60 \
             --speed-limit 1048576 --speed-time 300 -o "$part" "$url" \
             -w "     $name attempt $attempt: %{size_download} B at %{speed_download} B/s\n" \
          || sleep 30
        size=$(stat -c %s "$part" 2>/dev/null || echo 0)
    done

    if [ "$size" != "$remote" ]; then
        echo "FAIL $name: $size of $remote bytes after $attempt attempts"
        return 1
    fi
    if python unbound_gas/verify_snapshot_files.py install "$part" "$dest"; then
        echo "OK   $name: downloaded, verified, installed"
    else
        echo "FAIL $name: complete size but failed verification (staged copy kept at $part)"
        return 1
    fi
}
export -f fetch_one

# Task list: one "URL DEST AUTH" line per file, smallest datasets first so they
# become usable early.
TASKFILE=$(mktemp "../Outputs_Perlmutter/fetch_dmo-${SLURM_JOB_ID:-local}.tasks.XXXX")
trap 'rm -f "$TASKFILE"' EXIT
for ds in $DATASETS; do
    case $ds in
        simba)
            echo "$SIMBA_URL $DATA_ROOT/SIMBA/m100n1024/dm/snapshots/snap_m100n1024_009.hdf5 none" ;;
        illustris)
            for i in $(seq 0 127); do
                echo "$TNG_API/Illustris-1-Dark/files/snapshot-103.$i.hdf5" \
                     "$DATA_ROOT/IllustrisTNG/Illustris-1-Dark/output/snapdir_103/snap_103.$i.hdf5 tng"
            done ;;
        tng300)
            for i in $(seq 0 74); do
                echo "$TNG_API/TNG300-1-Dark/files/snapshot-67.$i.hdf5" \
                     "$DATA_ROOT/IllustrisTNG/TNG300-1-Dark/output/snapdir_067/snap_067.$i.hdf5 tng"
            done ;;
        *) echo "unknown dataset $ds" >&2; exit 1 ;;
    esac
done > "$TASKFILE"

echo "########## fetch DMO snapshots ##########"
echo "datasets : $DATASETS ($(wc -l < "$TASKFILE") files, $NPAR in parallel)"
echo "start    : $(date)"

xargs -P "$NPAR" -L 1 bash -c 'fetch_one "$0" "$1" "$2"' < "$TASKFILE"
fetch_rc=$?

echo
echo "########## snapshot completeness ##########"
check_rc=0
for ds in $DATASETS; do
    case $ds in
        simba)     pat="$DATA_ROOT/SIMBA/m100n1024/dm/snapshots/snap_m100n1024_009.hdf5" ;;
        illustris) pat="$DATA_ROOT/IllustrisTNG/Illustris-1-Dark/output/snapdir_103/snap_103.*.hdf5" ;;
        tng300)    pat="$DATA_ROOT/IllustrisTNG/TNG300-1-Dark/output/snapdir_067/snap_067.*.hdf5" ;;
    esac
    python unbound_gas/verify_snapshot_files.py snapshot "$pat" || check_rc=1
done

echo "finish   : $(date)"
echo "########## EXIT CODES: fetch=$fetch_rc completeness=$check_rc ##########"
[ "$fetch_rc" = 0 ] && [ "$check_rc" = 0 ]
