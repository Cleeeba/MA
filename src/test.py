import torch
import omegaconf
import sys

p = '/hkfs/work/workspace_haic/scratch/rx3495-workspace_C/outputs/2025-11-27/16-31-00-bdm1-qm9-qm9_no_h/checkpoints/qm9_no_h/epoch999.ckpt'

print("Registering OmegaConf DictConfig as safe global...")
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])

print("Trying torch.load with weights_only=False ...")
try:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    print("Load OK: type =", type(ck))
    if isinstance(ck, dict):
        print("Keys:", list(ck.keys())[:20])
    sys.exit(0)

except Exception as e:
    print("Load failed:", repr(e))
    sys.exit(1)
