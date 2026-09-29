#! /bin/bash -l

# Download the z ~ 0.75 and z = 1.0 snapshots for the cross_corr runs:
# FLAMINGO L1_m9 / fgas-8sigma / Jet_fgas-4sigma and L1_m9_DMO at snapshots 57
# (z = 1.00) and 62 (z = 0.75), TNG300-1 at 57 (z = 0.7574, mini) and 50
# (z = 0.9973, mini field list + NeutralHydrogenAbundance as a field subset),
# TNG300-1-Dark at 57 (mini) and 50 (full), with every group/SOAP catalogue.
# What and why: cross_corr/fetch_snapshot_tools.py (module docstring).
#
# Reads a manifest written beforehand by
#   python cross_corr/fetch_snapshot_tools.py manifest ../Outputs_Perlmutter/fetch_z075_z10_manifest.tsv
# (one line per file: dataset component url dest size resume auth kind).
#
# Per file: an existing destination is checked and skipped if it verifies; if
# it does not verify it is reported and left untouched (never overwritten).
# Otherwise the file is downloaded into a staging tree on the same filesystem,
# size-checked against the listing / Content-Length, verified
# (fetch_snapshot_tools.py check), and moved onto its destination with an
# atomic rename that refuses to replace an existing file. Durham files and the
# TNG snapshot-50 subset cannot be resumed (the servers ignore range requests),
# so a failed attempt restarts that file from zero; whole TNG files resume with
# curl -C -. Finally, unless CHECK_COMPLETE=0, each dataset is checked for
# completeness (all files present, particle and catalogue totals, FLAMINGO
# virtual-file sources and membership/X-ray row counts).
#
# The TNG API key is read from a header file (default ~/.config/tng/api_header,
# "api-key: <key>") via curl -H @file, so it never appears in `ps` or the logs.
#
# xfer QOS: free, runs on login nodes, meant for data staging (48 h cap); it
# must be paired with -C cron. Safe to resubmit: verified files are skipped.
#
# Submit from the scripts/ directory, one job per server, with disjoint
# DATASETS (two jobs working on the same files would race), e.g.:
#   DATASETS="flam_L1_m9_57 ..." sbatch --time=24:00:00 cross_corr/fetch_snapshots_z075_z10.sh
#   DATASETS="tng300_57 tng300_50 tngdark_57 tngdark_50" sbatch cross_corr/fetch_snapshots_z075_z10.sh
# Smoke test (one file of each kind, no completeness check):
#   MANIFEST=../Outputs_Perlmutter/fetch_z075_z10_smoke.tsv CHECK_COMPLETE=0 \
#       sbatch --time=03:00:00 cross_corr/fetch_snapshots_z075_z10.sh

#SBATCH -A desi
#SBATCH --qos=xfer
#SBATCH -C cron
#SBATCH --time=48:00:00
#SBATCH --job-name=fetch_z075_z10
#SBATCH -o ../Outputs_Perlmutter/fetch_z075_z10-%j.out

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

# The cosmodesi environment prepends a conda libcurl to LD_LIBRARY_PATH that is
# older than the system curl expects, so curl runs with the pre-activation
# library path; only the Python checks need the environment.
export SYS_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

sys_curl () {
    LD_LIBRARY_PATH="$SYS_LD_LIBRARY_PATH" /usr/bin/curl "$@"
}
export -f sys_curl

DATA_ROOT=${SIMSTACK_DATA_ROOT:-/pscratch/sd/r/rhliu/simulations}
export DATA_ROOT=${DATA_ROOT%/}
export STAGE_ROOT=${STAGE_ROOT:-$DATA_ROOT/staging_fetch_z075_z10}
export TNG_API_HEADER=${TNG_API_HEADER:-$HOME/.config/tng/api_header}
MANIFEST=${MANIFEST:-../Outputs_Perlmutter/fetch_z075_z10_manifest.tsv}
FULL_MANIFEST=${FULL_MANIFEST:-../Outputs_Perlmutter/fetch_z075_z10_manifest.tsv}
DATASETS=${DATASETS:-$(cut -f1 "$MANIFEST" | uniq | tr '\n' ' ')}
NPAR=${NPAR:-4}                 # concurrent downloads
CHECK_COMPLETE=${CHECK_COMPLETE:-1}
export MAX_ATTEMPTS=${MAX_ATTEMPTS:-10}
export MIN_BYTES=1024           # smaller answers are API errors ("Invalid input.")

if [ ! -s "$MANIFEST" ]; then
    echo "manifest $MANIFEST missing or empty" >&2
    exit 1
fi
if [ ! -s "$TNG_API_HEADER" ]; then
    echo "TNG API header file $TNG_API_HEADER missing or empty" >&2
    exit 1
fi

# fetch_one URL DEST SIZE RESUME AUTH KIND
fetch_one () {
    local url=$1 dest=$2 size=$3 resume=$4 auth=$5 kind=$6
    local name part remote got attempt rc restart
    local -a hdr=() cont=()
    name=${dest#"$DATA_ROOT"/}
    part="$STAGE_ROOT/$name.part"
    [ "$auth" = "tng" ] && hdr=(-H "@$TNG_API_HEADER")

    if [ -e "$dest" ]; then
        if python cross_corr/fetch_snapshot_tools.py check "$dest" "$kind" > /dev/null 2>&1; then
            echo "OK   $name: already present and verified"
            return 0
        fi
        echo "FAIL $name: already present but fails verification; left untouched"
        return 1
    fi

    if [ "$size" -gt 0 ]; then
        remote=$size
    else
        # Final-hop Content-Length (the TNG API answers with a 302 to a data
        # server): reset at every status line so an earlier hop never survives.
        remote=$(sys_curl -sIL -m 120 "${hdr[@]}" "$url" | tr -d '\r' \
                 | awk '/^HTTP\// {n=""} tolower($1)=="content-length:" {n=$2} END {print n}')
    fi
    if ! [[ "$remote" =~ ^[0-9]+$ ]] || [ "$remote" -lt "$MIN_BYTES" ]; then
        echo "FAIL $name: bad Content-Length '$remote' (server error?)"
        return 1
    fi

    mkdir -p "$(dirname "$part")"
    # Pass 1 resumes any partial; if its full-size result fails verification
    # (e.g. a stale partial from an earlier job), pass 2 fetches it once more
    # from zero, overwriting the staged copy.
    for pass in 1 2; do
        got=$(stat -c %s "$part" 2>/dev/null || echo 0)
        restart=0
        [ "$got" -gt "$remote" ] && restart=1
        if [ "$pass" = 2 ]; then restart=1; got=-1; fi
        attempt=0
        while [ "$got" -ne "$remote" ] && [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
            attempt=$((attempt + 1))
            # Without -C curl truncates the staged file and starts from zero,
            # the only option on servers that ignore range requests. --fail
            # keeps an HTTP error body out of the staged file.
            if [ "$resume" = 1 ] && [ "$restart" = 0 ]; then cont=(-C -); else cont=(); fi
            sys_curl -sSL --fail "${hdr[@]}" "${cont[@]}" --connect-timeout 60 \
                 --speed-limit 1048576 --speed-time 300 -o "$part" "$url" \
                 -w "     $name attempt $attempt: HTTP %{http_code}, %{size_download} B at %{speed_download} B/s\n"
            rc=$?
            got=$(stat -c %s "$part" 2>/dev/null || echo 0)
            restart=0
            # 33: the server refused the range request; > remote: a stale or
            # appended partial. Both restart from zero on the next attempt.
            if [ "$rc" = 33 ] || [ "$got" -gt "$remote" ]; then restart=1; fi
            [ "$rc" != 0 ] && [ "$got" -ne "$remote" ] && sleep 30
        done

        if [ "$got" -ne "$remote" ]; then
            echo "FAIL $name: $got of $remote bytes after $attempt attempts"
            return 1
        fi
        if python cross_corr/fetch_snapshot_tools.py install "$part" "$dest" "$kind"; then
            echo "OK   $name: downloaded, verified, installed"
            return 0
        fi
        [ -e "$dest" ] && break   # another writer installed it; do not retry
        echo "     $name: full size but failed verification (pass $pass)"
    done
    echo "FAIL $name: not installed (staged copy kept at $part)"
    return 1
}
export -f fetch_one

echo "########## fetch z ~ 0.75 and z = 1.0 snapshots ##########"
echo "manifest : $MANIFEST"
echo "datasets : $DATASETS"
echo "staging  : $STAGE_ROOT"
echo "parallel : $NPAR"
echo "start    : $(date)"

for ds in $DATASETS; do
    if ! cut -f1 "$MANIFEST" | grep -qx -- "$ds"; then
        echo "dataset '$ds' is not in $MANIFEST" >&2
        exit 1
    fi
done

# Manifest lines of the selected datasets, fields 3-8, straight into xargs.
awk -F'\t' -v sel=" $DATASETS " 'index(sel, " " $1 " ") {print $3, $4, $5, $6, $7, $8}' "$MANIFEST" \
    | xargs -P "$NPAR" -L 1 bash -c 'fetch_one "$@"' _
fetch_rc=$?

check_rc=0
if [ "$CHECK_COMPLETE" = 1 ]; then
    echo
    echo "########## completeness ##########"
    python cross_corr/fetch_snapshot_tools.py complete "$FULL_MANIFEST" $DATASETS || check_rc=1
fi

echo "finish   : $(date)"
echo "########## EXIT CODES: fetch=$fetch_rc completeness=$check_rc ##########"
[ "$fetch_rc" = 0 ] && [ "$check_rc" = 0 ]
