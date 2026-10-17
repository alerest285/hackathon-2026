# Chat: saadl27 / saad-calib

Append-only. One line per message. Written by scripts/say.sh.

- 2026-10-17T19:17Z saadl27/saad-calib -> @alerest285: Saad asks: is a second GPU instance running besides Runpod rmhs4cu2cejemv (<pod-host>:<ssh-port>), on AWS eu-north-1 or a second pod? The logs only show that pod, and T03 (AWS H100) is still todo. Saad's laptop has no AWS CLI/profile and no Runpod key, so we can't check from here. Reason: we want to score joint maps on Saad's 82-episode dataset with the model (~16 GB eager, ~30 min) without slowing the qmap LoRA run that is starting on the pod. If one is free, please send host:port.
