"""Cloudflare R2 client for storing trained models."""

import os

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


class R2Client:
    def __init__(
        self,
        access_key: str = "",
        secret_key: str = "",
        endpoint: str = "",
        bucket: str = "",
    ):
        self.access_key = access_key or os.environ.get("SOMERSVC_R2_ACCESS_KEY", "")
        self.secret_key = secret_key or os.environ.get("SOMERSVC_R2_SECRET_KEY", "")
        self.endpoint = endpoint or os.environ.get("SOMERSVC_R2_ENDPOINT", "")
        self.bucket = bucket or os.environ.get("SOMERSVC_R2_BUCKET", "somersvc-models")

    def _client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )

    def is_configured(self) -> bool:
        return bool(self.access_key and self.secret_key and self.endpoint)

    def upload_file(self, local_path: str, r2_key: str):
        """Upload a local file to R2."""
        self._client().upload_file(local_path, self.bucket, r2_key)

    def download_file(self, r2_key: str, local_path: str, callback=None):
        """Download a file from R2 to local disk. `callback(bytes)` is invoked
        with bytes-transferred-this-chunk during download (boto3 convention)
        so callers can show progress.
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # Single-threaded transfer. boto3's default pool (up to 10 threads)
        # runs inside the GUI process and starved the main thread of the
        # GIL during the download — the spinner and best-match GIF visibly
        # stuttered. One stream is plenty fast for R2 and keeps the UI smooth.
        self._client().download_file(
            self.bucket, r2_key, local_path, Callback=callback,
            Config=TransferConfig(use_threads=False),
        )

    def head_size(self, r2_key: str) -> int:
        """Total size of an object in bytes (0 if it doesn't exist)."""
        try:
            resp = self._client().head_object(Bucket=self.bucket, Key=r2_key)
            return int(resp.get("ContentLength", 0))
        except Exception:
            return 0

    def list_files(self, prefix: str) -> list[str]:
        """List files in R2 under a prefix."""
        resp = self._client().list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]

    def file_exists(self, r2_key: str) -> bool:
        """Check if a file exists in R2."""
        try:
            self._client().head_object(Bucket=self.bucket, Key=r2_key)
            return True
        except Exception:
            return False

    def delete_files(self, keys: list[str]):
        """Delete files from R2."""
        if not keys:
            return
        objects = [{"Key": k} for k in keys]
        self._client().delete_objects(
            Bucket=self.bucket, Delete={"Objects": objects}
        )

    def get_upload_script(self) -> str:
        """Return a Python script that can be run on the pod to upload model to R2."""
        return f'''
import boto3, glob, os
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url="{self.endpoint}",
    aws_access_key_id="{self.access_key}",
    aws_secret_access_key="{self.secret_key}",
    config=Config(signature_version="s3v4"),
    region_name="auto",
)

bucket = "{self.bucket}"
job_id = os.environ["SVC_JOB_ID"]
speaker = os.environ["SVC_SPEAKER"]

# Find latest checkpoint
g_files = sorted(glob.glob("/workspace/logs/44k/G_*.pth"))
if not g_files:
    print("ERROR: No checkpoint found")
    exit(1)

latest = g_files[-1]
g_name = os.path.basename(latest)
prefix = f"models/{{speaker}}/{{job_id}}"

print(f"Uploading {{g_name}} to R2...")
s3.upload_file(latest, bucket, f"{{prefix}}/{{g_name}}")
print(f"Uploaded {{g_name}}")

# Upload config
config_path = "/workspace/configs/44k/config.json"
if os.path.exists(config_path):
    s3.upload_file(config_path, bucket, f"{{prefix}}/config.json")
    print("Uploaded config.json")

# Write a marker file so app knows upload is complete
import tempfile, json
marker = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
json.dump({{"checkpoint": g_name, "status": "ready"}}, marker)
marker.close()
s3.upload_file(marker.name, bucket, f"{{prefix}}/_complete.json")
os.unlink(marker.name)
print("Upload complete — marker written")
'''
