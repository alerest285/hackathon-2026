#!/bin/bash
# EC2 first-boot script (runs on the AWS GPU machine, never locally).
# Gives every member in members.txt ssh access via their GitHub keys, installs ffmpeg,
# and arms the idle auto-stop that protects the budget.
set -eux
for u in __MEMBERS__; do [ "$u" = none ] && continue; curl -fsSL "https://github.com/$u.keys" >> /home/ubuntu/.ssh/authorized_keys || true; done
sort -u -o /home/ubuntu/.ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
chown ubuntu:ubuntu /home/ubuntu/.ssh/authorized_keys; chmod 600 /home/ubuntu/.ssh/authorized_keys
apt-get update -y && apt-get install -y ffmpeg tmux git
cat > /usr/local/bin/idle-stop.sh <<'IDLE'
#!/bin/bash
# Stops this instance after __IDLE_MIN__ minutes with no SSH connection and an idle GPU.
# Persistent spot + stop behaviour: a guest power-off = EC2 "stopped", disk kept.
# Disable with: sudo touch /etc/no-idle-stop
[ -f /etc/no-idle-stop ] && exit 0
f=/var/tmp/idle-since; busy=0
ss -Htn state established '( sport = :22 )' | grep -q . && busy=1
u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
[ "${u:-0}" -gt 5 ] && busy=1
if [ $busy = 1 ] || [ ! -f $f ]; then date +%s > $f; exit 0; fi
[ $(( $(date +%s) - $(cat $f) )) -gt $(( __IDLE_MIN__ * 60 )) ] && shutdown -h now "idle stop"
IDLE
chmod +x /usr/local/bin/idle-stop.sh
echo '*/5 * * * * root /usr/local/bin/idle-stop.sh' > /etc/cron.d/idle-stop
