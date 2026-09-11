#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/jlens/results /run/sshd /root/.ssh
apt-get update > /workspace/jlens/results/system_setup.log 2>&1
apt-get install -y --no-install-recommends openssh-server rsync ca-certificates libgomp1 >> /workspace/jlens/results/system_setup.log 2>&1
python - <<'PY'
import os, pathlib, re
key = os.environ['PUBLIC_KEY'].strip()
name = os.environ.get('JLENS_LEASE_NAME')
if name:
    if not re.fullmatch(r'jlens-confirm-[a-f0-9]{32}', name):
        raise SystemExit('invalid lease name')
    pathlib.Path('/workspace/jlens/provider_lease_name').write_text(name + '\n')
if not re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,3}(?: [^\r\n]*)?', key):
    raise SystemExit('invalid SSH public key')
directory = pathlib.Path('/root/.ssh')
directory.chmod(0o700)
path = directory / 'authorized_keys'
path.write_text(key + '\n')
path.chmod(0o600)
pathlib.Path('/etc/ssh/sshd_config.d/jlens.conf').write_text(
    'PermitRootLogin prohibit-password\nPasswordAuthentication no\n'
    'KbdInteractiveAuthentication no\nAllowAgentForwarding no\n'
    'AllowTcpForwarding no\nX11Forwarding no\n')
PY
unset PUBLIC_KEY
/usr/sbin/sshd -t
exec /usr/sbin/sshd -D -e
