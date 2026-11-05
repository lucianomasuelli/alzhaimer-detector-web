#!/bin/bash
#SBATCH --job-name=e1-export-2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/lmasuelli/alzhaimer-detector-web/out_e1_early/logs/job2_%j.out
#SBATCH --error=/home/lmasuelli/alzhaimer-detector-web/out_e1_early/logs/job2_%j.err
source /home/lmasuelli/alzhaimer-detector-web/out_e1_early/tools/env_header.sh
set +e
D="$RES/dataset/Ivanova/train"
EMB="$WEB/out_e1_early/logs/emb"; mkdir -p "$EMB"
for spec in AD/AD-M-82-321.mp3 AD/AD-M-57-163.wav HC/HC-M-59-291.mp3 HC/HC-M-65-48.wav HC/HC-M-84-231.wav; do
  g=${spec%%/*}; f=${spec#*/}; uid=${f%.*}
  A="$D/audio/$spec"; R="$D/text/$g/${uid}_pooledaudio_audioxlsr1b.pt"
  echo "=== $uid ($g) ==="; ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -of csv=p=0 "$A"
  echo "--- GPU (tools/parity_device.py --device cuda)"; python out_e1_early/tools/parity_device.py "$A" "$R" --device cuda --save "$EMB/${uid}_cuda.pt"; echo "exit=$?"
  echo "--- CPU (scripts/check_parity.py, sin cambios)"; python scripts/check_parity.py "$A" "$R"; echo "exit=$?"
  echo "--- CPU (tools/parity_device.py --device cpu, para comparar GPU vs CPU)"; python out_e1_early/tools/parity_device.py "$A" "$R" --device cpu --save "$EMB/${uid}_cpu.pt"; echo "exit=$?"
  python -c "
import torch; a=torch.load('$EMB/${uid}_cuda.pt'); b=torch.load('$EMB/${uid}_cpu.pt'); d=(a-b).abs()
print(f'GPU vs CPU: max|diff|={d.max():.3e} cosine={torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0).item():.8f}')"
done
