#!/bin/bash
# Stage 2.8 active-babysit poll: auto-relaunch dead arms, wait ~105 s
# (keeps the container computing during the active turn), report state.
# Usage: bash babysit_poll.sh [TARGET_PHI]   (default 0.85)
cd /home/user/SPARK/stage21
TP=${1:-0.85}
relaunch () {  # tag raise seed
  if [ ! -f "stage28_${1}_result.json" ] && ! pgrep -f "tag ${1}" >/dev/null; then
    nohup python3 run_stage28.py --raise-oxide "${2}" --raise-scope CO \
      --target-phi "${TP}" --seed "${3}" --tag "${1}" \
      > "stage28_${1}.log" 2>&1 &
    echo "RELAUNCHED ${1}"
  fi
}
relaunch r050s1 0.5 1
relaunch r050s2 0.5 2
relaunch r060s1 0.6 1
relaunch r060s2 0.6 2
end=$((SECONDS+${2:-105}))
while [ $SECONDS -lt $end ]; do
  n=$(ls stage28_r0*_result.json 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && break
  sleep 5
done
for t in r050s1 r050s2 r060s1 r060s2; do tail -1 "stage28_${t}.log"; done
echo "results: $(ls stage28_*_result.json 2>/dev/null | tr '\n' ' ')"
echo "uptime: $(cut -d' ' -f1 /proc/uptime)  procs: $(pgrep -fc run_stage28)"
