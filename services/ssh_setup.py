"""Auto-generate SSH keys and register them with RunPod for new users."""

import os
import subprocess
from pathlib import Path

import requests


SSH_KEY_PATH = Path.home() / ".ssh" / "somersvc_rsa"
SSH_PUB_PATH = Path.home() / ".ssh" / "somersvc_rsa.pub"


def ensure_ssh_key() -> tuple[str, str]:
    """Generate an SSH keypair at ~/.ssh/somersvc_rsa if missing.

    Returns (private_path, public_key_text).
    """
    SSH_KEY_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    if not SSH_KEY_PATH.exists():
        # Use ssh-keygen for compatibility with paramiko/openssh formats
        subprocess.run(
            [
                "ssh-keygen",
                "-t", "rsa",
                "-b", "4096",
                "-f", str(SSH_KEY_PATH),
                "-N", "",  # no passphrase
                "-C", "somersvc",
            ],
            check=True,
            capture_output=True,
        )
        # Tighten perms
        os.chmod(SSH_KEY_PATH, 0o600)
        os.chmod(SSH_PUB_PATH, 0o644)

    pub_text = SSH_PUB_PATH.read_text().strip()
    return str(SSH_KEY_PATH), pub_text


def register_public_key_with_runpod(api_key: str, public_key: str) -> tuple[bool, str]:
    """Upload the public key to the user's RunPod account.

    RunPod stores ONE public key on the account; this overwrites it.
    Returns (success, message).
    """
    if not api_key or not public_key:
        return False, "Missing API key or public key"

    # RunPod GraphQL mutation to save user settings (replaces public key)
    query = """
    mutation SetSshKey($input: PodEditJobInput!) {
        updateUserSettings(input: $input) {
            id
        }
    }
    """
    # Actual RunPod mutation for user settings:
    mutation = {
        "query": (
            "mutation SaveUserSettings($input: UpdateUserSettingsInput!) {"
            "  updateUserSettings(input: $input) { pubKey }"
            "}"
        ),
        "variables": {"input": {"pubKey": public_key}},
    }

    try:
        resp = requests.post(
            f"https://api.runpod.io/graphql?api_key={api_key}",
            json=mutation,
            timeout=15,
        )
        data = resp.json()
        if data.get("errors"):
            err_msg = data["errors"][0].get("message", "unknown error")
            return False, err_msg
        return True, "Public key uploaded to RunPod"
    except requests.RequestException as e:
        return False, f"Network error: {e}"
    except Exception as e:
        return False, f"Error: {e}"
