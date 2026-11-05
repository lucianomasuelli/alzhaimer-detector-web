# Entorno comun para los jobs de exportacion (mismo estilo que submit_mendieta_e1_*.sh)
set -e -o pipefail
CONDA_PARENT="/home/lmasuelli/miniconda3"
ENV_PATH="${CONDA_PARENT}/envs/cogni"
source "${CONDA_PARENT}/etc/profile.d/conda.sh"
conda activate cogni
export WANDB_MODE=disabled WANDB_SILENT=true TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="${ENV_PATH}/lib/python3.10/site-packages:$HOME/dementiaDetection/Code/alzhaimer-detector/modules:$PYTHONPATH"
export PATH="$PATH:${CONDA_PARENT}/envs/ffmpeg/bin"   # ffmpeg/ffprobe del entorno aparte (al final: no tapa python de cogni)
WEB="$HOME/alzhaimer-detector-web"
RES="$HOME/dementiaDetection/Code/alzhaimer-detector"
cd "$WEB"
