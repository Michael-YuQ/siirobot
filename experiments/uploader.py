"""
结果上传器 — 定期将 checkpoint 和日志上传到网盘服务器

服务器 API:
    POST /upload   multipart/form-data, fields: file + path(子目录)
    POST /mkdir    JSON body: {"path": "xxx"}
    GET  /list     ?path=xxx
"""
import os
import json
import tarfile
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime


UPLOAD_BASE = "http://111.170.6.103:10005"


class ResultUploader:
    """
    训练结果上传器

    上传到: {remote_root}/{experiment_id}/
    例如:   atpc_experiments/atpc-G1-seed42/

    用法:
        uploader = ResultUploader(experiment_id="atpc-G1-seed42")
        uploader.upload_checkpoint(log_dir, iteration=500)
    """

    def __init__(
        self,
        experiment_id: str = "unknown",
        remote_root: str = "atpc_experiments",
        base_url: str = UPLOAD_BASE,
    ):
        self.base_url = base_url.rstrip("/")
        self.experiment_id = experiment_id
        self.remote_root = remote_root
        self.remote_dir = f"{remote_root}/{experiment_id}"
        self.upload_count = 0
        self._ensure_remote_dir()

    def _ensure_remote_dir(self):
        """在服务器上创建实验目录"""
        try:
            import requests
            # 先创建根目录
            requests.post(
                f"{self.base_url}/mkdir",
                json={"path": self.remote_root},
                timeout=10,
            )
            # 再创建实验子目录
            requests.post(
                f"{self.base_url}/mkdir",
                json={"path": self.remote_dir},
                timeout=10,
            )
        except Exception as e:
            print(f"[Uploader] mkdir warning: {e}")

    def upload_file(self, filepath: str, remote_subdir: str = None):
        """上传单个文件"""
        try:
            import requests
            target_path = remote_subdir or self.remote_dir
            with open(filepath, "rb") as f:
                resp = requests.post(
                    f"{self.base_url}/upload",
                    files={"file": (os.path.basename(filepath), f)},
                    data={"path": target_path},
                    timeout=300,
                )
            if resp.status_code == 200:
                print(f"[Uploader] OK: {os.path.basename(filepath)} -> {target_path}")
                return True
            else:
                print(f"[Uploader] Failed ({resp.status_code}): {resp.text[:200]}")
                return False
        except Exception as e:
            print(f"[Uploader] Error: {e}")
            return False

    def upload_checkpoint(self, log_dir: str, iteration: int):
        """上传当前 checkpoint + 训练日志 (打包为 tar.gz)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{self.experiment_id}_iter{iteration}_{timestamp}.tar.gz"

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, archive_name)
            self._create_archive(log_dir, archive_path, iteration)
            self.upload_file(archive_path)

        self.upload_count += 1
        print(f"[Uploader] Checkpoint #{self.upload_count}: iter {iteration}")

    def upload_final(self, log_dir: str):
        """训练结束后上传完整结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{self.experiment_id}_FINAL_{timestamp}.tar.gz"

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, archive_name)
            self._create_full_archive(log_dir, archive_path)
            self.upload_file(archive_path)

        print(f"[Uploader] Final upload complete: {archive_name}")

    def _create_archive(self, log_dir, archive_path, iteration):
        """创建增量 checkpoint 压缩包"""
        with tarfile.open(archive_path, "w:gz") as tar:
            for name in [
                f"model_{iteration}.pt",
                "training_stats.json",
                "adversarial_state.pt",
                "experiment_config.json",
            ]:
                fpath = os.path.join(log_dir, name)
                if os.path.exists(fpath):
                    tar.add(fpath, arcname=name)

    def _create_full_archive(self, log_dir, archive_path):
        """创建完整结果压缩包"""
        with tarfile.open(archive_path, "w:gz") as tar:
            for f in os.listdir(log_dir):
                fpath = os.path.join(log_dir, f)
                if os.path.isfile(fpath):
                    tar.add(fpath, arcname=f)
