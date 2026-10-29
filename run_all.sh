#!/usr/bin/env bash
# Run the analysis pipeline end to end. Each step writes a log to results/logs/<step>.log.
#
#   bash run_all.sh data           steps 1-4, 4b, 13, 15: corpora -> cohort -> windows and annotation
#                                  requests; validation datasets and the request files of the local
#                                  validation kit; lexicon features; cohort description
#   bash run_all.sh main           steps 5-12 and the sensitivity analyses on the stored ratings of the
#                                  main run (llama31_8b), then 14, 14b-14e, 18 (when their inputs
#                                  exist), the course types (20) and the figures and numbers (21-23)
#   bash run_all.sh analyse <run>  steps 6-12 and 20 for one parsed annotation run
#   bash run_all.sh figures        steps 21-23 only (figures, Table 1, numbers quoted in the text)
#   bash run_all.sh synthetic      pipeline test on a simulated stand-in for the model outputs
#                                  (needs the windows of step 3; uses reduced replicate counts)
#
# Options (environment variables):
#   QUICK=1    reduced numbers of initializations, bootstrap resamples and shuffles (tests only)
#   NJOBS=4    parallel workers for the NMF, DPMM and EM restarts (results do not depend on it)
#   RESUME=1   skip steps whose log ends with STEP_OK (finished in an earlier call)
#
# Run times on two CPU cores with the full settings: data about 10 min; main about 11 h, of which the
# DPMM sweeps of step 10 take about 3.5 h, the NMF stability sweeps of step 6 about 2.5 h, the course
# types of step 20 about 1 h and the predictive validation of step 14d about 1 h per annotating model.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
ROOT="$(dirname "$SELF")"
cd "$ROOT/scripts"
NJ=${NJOBS:-2}
LOG="$ROOT/results/logs"; mkdir -p "$LOG"
MAIN=llama31_8b

if [ "${QUICK:-0}" = 1 ]; then
  NMF="--reps 5 --reps-final 10"; FINE="--reps 5"; BROAD="--reps 3"; PSY="--n-boot 20"
  VAR="--n-boot 50"; DP="--n-init 1"; REG="--n-init-sweep 20 --n-init-final 100 --n-boot 20"
  ROB="--n-splits 3 --n-shuffle 5"; ROB8="--n-shuffle 5"; VAL="--n-boot 100"
  CRS="--k-max 3 --n-starts 2 --n-starts-cv 1 --n-null 2 --n-shuffle 3"
else
  NMF="--reps 20 --reps-final 100"; FINE="--reps 100"; BROAD="--reps 20"; PSY="--n-boot 200"
  VAR="--n-boot 500"; DP="--n-init 10"; REG="--n-init-sweep 1000 --n-init-final 10000 --n-boot 200"
  ROB="--n-splits 20 --n-shuffle 200"; ROB8="--n-shuffle 200"; VAL="--n-boot 1000"
  CRS="--k-max 6 --n-starts 10 --n-starts-cv 4 --n-null 10 --n-shuffle 20"
fi

step () {  # step <log name> <command ...>: run one pipeline step and record its log
  local name=$1; shift
  if [ "${RESUME:-0}" = 1 ] && [ -f "$LOG/$name.log" ] && [ "$(tail -n 1 "$LOG/$name.log")" = STEP_OK ]; then
    echo "[$(date +%H:%M:%S)] $name (finished earlier, skipped)"; return 0
  fi
  echo "[$(date +%H:%M:%S)] $name"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED: $name (see results/logs/$name.log)"; exit 1; }
  echo "elapsed_s $((SECONDS - t0))" >> "$LOG/$name.log"; echo STEP_OK >> "$LOG/$name.log"
}

analyse () {  # analyse <run>: factors, psychometrics, dynamics, states, regimes, robustness
  local r=$1
  step "06_nmf_$r"        python3 06_nmf.py --run "$r" $NMF --sweep-ks none --n-jobs "$NJ"
  step "07_psy_$r"        python3 07_psychometrics.py --run "$r" $PSY
  step "08_var_$r"        python3 08_var_population.py --run "$r" $VAR
  step "09_umap_$r"       python3 09_umap.py --run "$r"
  step "10_dpmm_$r"       python3 10_states_dpmm.py --run "$r" $DP --n-jobs "$NJ"
  step "10c_names_$r"     python3 10c_name_states.py --run "$r"
  step "11_regimes_$r"    python3 11_regimes.py --run "$r" $REG --n-jobs "$NJ"
  step "12_robust_$r"     python3 12_robustness.py --run "$r" $ROB
  step "12c_community_$r" python3 12c_community_replication.py --run "$r"
}

course () {  # course <run>: course types over the year (uses the inventories of step 18 when they exist)
  step "20_course_$1"     python3 20_course_types.py --run "$1" $CRS --n-jobs "$NJ"
}

sensitivity () {  # sensitivity <run>: NMF stability sweeps, eight states, K = 10 and K = 18
  local r=$1
  # staNMF instability: fine sweep around K = 12 and broad sweep over K = 3-25 (Fig. 2b)
  step "06_nmf_${r}_fine"    python3 06_nmf.py --run "$r" --sweep-only --sweep-ks 11,13,14 $FINE --tag _fine \
                             --n-jobs "$NJ"
  step "06_nmf_${r}_broad"   python3 06_nmf.py --run "$r" --sweep-only --k-min 3 --k-max 25 $BROAD \
                             --tag _broad --n-jobs "$NJ"
  # eight states: the adjacent, more regularized step of the DPMM sweep
  step "10b_states_${r}_8states" python3 10b_alt_states.py --run "$r" --rule adjacent --tag _8states
  step "10c_names_${r}_8states"  python3 10c_name_states.py --run "$r" --tag _8states
  step "11_regimes_${r}_8states" python3 11_regimes.py --run "$r" $REG --n-jobs "$NJ" --tag _8states
  step "12_robust_${r}_8states"  python3 12_robustness.py --run "$r" --parts D,F,E $ROB8 --tag _8states
  # K = 10 (Extended Data Fig. 5e) and the K = 18 solution with its downstream models
  step "06_nmf_${r}_k10"    python3 06_nmf.py --run "$r" --k 10 --reps-final 10 --sweep-ks none --tag _k10
  step "06_nmf_${r}_k18"    python3 06_nmf.py --run "$r" --k 18 --reps-final 100 --sweep-ks none --tag _k18 \
                            --names ../config/factor_names_k18.yaml --n-jobs "$NJ"
  step "06_nmf_${r}_k18_fine" python3 06_nmf.py --run "$r" --sweep-only --sweep-ks 17,19 $FINE \
                             --tag _k18_fine --n-jobs "$NJ"
  step "07_psy_${r}_k18"     python3 07_psychometrics.py --run "$r" $PSY --tag _k18
  step "08_var_${r}_k18"     python3 08_var_population.py --run "$r" $VAR --tag _k18
  step "12C_split_${r}_k18"  python3 12_robustness.py --run "$r" --parts C $ROB --tag _k18
  step "10_dpmm_${r}_k18"    python3 10_states_dpmm.py --run "$r" $DP --n-jobs "$NJ" --tag _k18
  step "10c_names_${r}_k18"  python3 10c_name_states.py --run "$r" --tag _k18
  step "11_regimes_${r}_k18" python3 11_regimes.py --run "$r" $REG --n-jobs "$NJ" --tag _k18
  step "12_robust_${r}_k18"  python3 12_robustness.py --run "$r" --parts D,E,F $ROB8 --tag _k18
}

figures () {  # figures, Table 1 and the numbers quoted in the text
  step 21_figures_main      python3 21_figures_main.py
  step 22_figures_extended  python3 22_figures_extended.py
  step 23_numbers           python3 23_manuscript_numbers.py
}

case "${1:-help}" in
  data)
    step 01_ingest          python3 01_ingest_corpus.py
    step 02_cohort          python3 02_select_cohort.py
    step 03_windows         python3 03_build_windows.py
    step 04_validation_sets python3 04_prepare_validation.py
    step 04b_local_requests python3 04b_prepare_local_validation.py
    step 13_lexicon         python3 13_lexicon_features.py
    step 15_cohort          python3 15_cohort_description.py ;;
  main)
    step "05_parse_$MAIN" python3 05_parse_collate.py --run "$MAIN"
    analyse "$MAIN"
    sensitivity "$MAIN"
    step 14_validation      python3 14_validation_analysis.py --main-run "$MAIN" $VAL
    if ls "$ROOT"/local_validation/outputs/*.jsonl > /dev/null 2>&1; then
      step 14b_validation_local     python3 14b_validation_local.py
      step 14c_model_comparison     python3 14c_model_comparison_local.py
      step 14d_validation_predictive python3 14d_validation_predictive.py
      step 14e_single_symptom       python3 14e_validation_single_symptom.py
    fi
    if [ -d "$ROOT/data/llm_outputs/llm_export/7f390e2d-314e-4023-b3af-0800aa9ecc1a" ] || \
       [ -d "$ROOT/data/llm_outputs/7f390e2d-314e-4023-b3af-0800aa9ecc1a" ]; then
      step 18_inventories           python3 18_inventories.py
    fi
    course "$MAIN"
    figures ;;
  analyse)
    analyse "${2:?annotation run, e.g. llama31_8b}"
    course "$2" ;;
  figures)
    figures ;;
  synthetic)
    QUICK=1
    NMF="--reps 5 --reps-final 10"; PSY="--n-boot 20"; VAR="--n-boot 50"; DP="--n-init 1"
    REG="--n-init-sweep 20 --n-init-final 100 --n-boot 20"; ROB="--n-splits 3 --n-shuffle 5"
    step 00_synthetic_export python3 ../tests/make_synthetic_llm_export.py
    CRS="--k-max 3 --n-starts 2 --n-starts-cv 1 --n-null 2 --n-shuffle 3"
    step 05_parse_synthetic  python3 05_parse_collate.py --run synthetic
    analyse synthetic
    course synthetic ;;
  *) sed -n 2,22p "$SELF" ;;
esac
