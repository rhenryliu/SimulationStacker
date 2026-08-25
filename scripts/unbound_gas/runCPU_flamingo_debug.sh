#! /bin/bash -l

# Fast validation of the FLAMINGO precompute path, for shaking out wiring
# errors without waiting on the 12 h production array (runCPU_flamingo_masked.sh).
#
# Runs the REAL code path -- same driver, same stacker, same mapMaker -- but:
#   --max-chunks 1   read 1 of the 64 snapshot chunk files instead of all
#   --pixel-size 2.0 nPixels 887 instead of 3548 (so no filename can collide
#                    with a production product)
#   --no-save        writes nothing; --max-chunks refuses to run without it,
#                    and save_data is stubbed out so even the cubic
#                    intermediate that makeField would otherwise write is
#                    suppressed
#
# The fields it produces are physically meaningless. The only question it
# answers is "does every step execute without raising".
#
# Part 1 covers all six (variant, pType) pairs unmasked -- i.e. every index of
# the production array. Part 2 takes one pair through the masked path at two
# radii, exercising create_masked_field, the halo selection and the projection.
#
# Submit from the scripts/ directory:
#   sbatch unbound_gas/runCPU_flamingo_debug.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -o ../Outputs_Perlmutter/flamingo_debug-%j.out # STDOUT
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a
# line starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless,
# but it reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

COMMON="--max-chunks 1 --pixel-size 2.0 --no-save"
rc=0

echo "########## Part 1: all six production array indices, unmasked ##########"
for FEEDBACK in L1_m9 fgas-8sigma Jet_fgas-4sigma; do
  for PTYPE in tSZ tau; do
    echo "---------- $FEEDBACK / $PTYPE ----------"
    srun --cpu-bind=cores python -u unbound_gas/precompute_flamingo_masked.py \
      --feedback "$FEEDBACK" --ptype "$PTYPE" --mask-radii $COMMON || rc=1
  done
done

echo "########## Part 2: masked path, one pair, two radii ##########"
srun --cpu-bind=cores python -u unbound_gas/precompute_flamingo_masked.py \
  --feedback L1_m9 --ptype tSZ --skip-unmasked --mask-radii 1.0 3.0 $COMMON || rc=1

if [ $rc -eq 0 ]; then
  echo "########## VALIDATION PASSED ##########"
else
  echo "########## VALIDATION FAILED ##########"
fi
exit $rc
