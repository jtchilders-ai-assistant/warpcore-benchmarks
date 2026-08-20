#!/usr/bin/env bash
export PATH=$HOME/pi30-bench/node/bin:$PATH
cd $HOME/pi30-bench/pi-30/harness
rm -f $HOME/pi30-bench/PI30_ORNITH_DONE
PI_TIMEOUT=600 bash run_model_30.warpcore.sh warpcore "ornith-ai/Ornith-1.0-35B-FP8" ornith > $HOME/pi30-bench/pi30_ornith.log 2>&1
echo "SCORE_LINE: $(grep -h "^SCORE" runs30/ornith/SUMMARY.txt 2>/dev/null)" >> $HOME/pi30-bench/pi30_ornith.log
touch $HOME/pi30-bench/PI30_ORNITH_DONE
