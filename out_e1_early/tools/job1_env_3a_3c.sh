#!/bin/bash
#SBATCH --job-name=e1-export-1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/lmasuelli/alzhaimer-detector-web/out_e1_early/logs/job1_%j.out
#SBATCH --error=/home/lmasuelli/alzhaimer-detector-web/out_e1_early/logs/job1_%j.err
source /home/lmasuelli/alzhaimer-detector-web/out_e1_early/tools/env_header.sh
echo "== 2. entorno =="; python out_e1_early/tools/env_info.py
echo "== 3a =="; python scripts/check_head_vs_research.py "$RES" || echo "3a EXIT=$?"
echo "== 3c =="; python scripts/audio_durations.py "$RES/dataset/Ivanova/train/audio" --csv durations.csv || echo "3c EXIT=$?"
