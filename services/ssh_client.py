"""Paramiko-based SSH/SFTP client for remote command execution and file transfer."""

import os
import stat
import tarfile
import tempfile
from pathlib import Path
from typing import Callable

import paramiko


class SSHClient:
    def __init__(self):
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    def connect(self, host: str, port: int, key_path: str, username: str = "root"):
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        pkey = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
        self._client.connect(
            hostname=host,
            port=port,
            username=username,
            pkey=pkey,
            timeout=30,
        )
        self._sftp = self._client.open_sftp()

    def exec_command(
        self,
        cmd: str,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> int:
        if not self._client:
            raise RuntimeError("Not connected")

        _, stdout, stderr = self._client.exec_command(cmd, get_pty=True)

        # Stream stdout line by line
        for line in iter(stdout.readline, ""):
            if on_stdout:
                on_stdout(line.rstrip("\n"))

        # Read any remaining stderr
        for line in iter(stderr.readline, ""):
            if on_stderr:
                on_stderr(line.rstrip("\n"))

        return stdout.channel.recv_exit_status()

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        progress_cb: Callable[[int, int], None] | None = None,
    ):
        if not self._sftp:
            raise RuntimeError("Not connected")

        def _progress(transferred: int, total: int):
            if progress_cb:
                progress_cb(transferred, total)

        self._sftp.put(local_path, remote_path, callback=_progress)

    def download_file(
        self,
        remote_path: str,
        local_path: str,
        progress_cb: Callable[[int, int], None] | None = None,
    ):
        if not self._sftp:
            raise RuntimeError("Not connected")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        def _progress(transferred: int, total: int):
            if progress_cb:
                progress_cb(transferred, total)

        self._sftp.get(remote_path, local_path, callback=_progress)

    def upload_directory(self, local_dir: str, remote_path: str):
        """Tar a local directory and upload it, then extract on remote."""
        if not self._sftp:
            raise RuntimeError("Not connected")

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(local_dir, arcname=os.path.basename(local_dir))

            remote_tar = f"{remote_path}/upload.tar.gz"
            self._sftp.put(tmp_path, remote_tar)
            self.exec_command(f"cd {remote_path} && tar xzf upload.tar.gz && rm upload.tar.gz")
        finally:
            os.unlink(tmp_path)

    def list_remote_files(self, remote_dir: str) -> list[str]:
        if not self._sftp:
            raise RuntimeError("Not connected")
        try:
            return self._sftp.listdir(remote_dir)
        except FileNotFoundError:
            return []

    def close(self):
        if self._sftp:
            self._sftp.close()
            self._sftp = None
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
