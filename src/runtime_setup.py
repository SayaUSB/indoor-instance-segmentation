import os
from pathlib import Path
HERE=Path(__file__).resolve().parents[1] / 'work'
os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime_config')
os.environ['TORCH_HOME']=str(HERE/'weights')
os.environ['MPLCONFIGDIR']=str(HERE/'runtime_config'/'matplotlib')
os.environ['WANDB_MODE']='disabled'
