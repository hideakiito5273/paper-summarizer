"""ユーザー管理 (.state/users.json) とパスワード検証。外部サービスや追加ライブラリは使わない。"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path

USER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,31}$")
_N, _R, _P = 2**14, 8, 1  # scrypt パラメータ


def _users_path(state_dir: Path) -> Path:
    return state_dir / "users.json"


def load_users(state_dir: Path) -> dict:
    try:
        return json.loads(_users_path(state_dir).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _save(state_dir: Path, users: dict) -> None:
    path = _users_path(state_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32).hex()


def set_password(state_dir: Path, username: str, password: str, display_name: str | None = None) -> None:
    if not USER_RE.match(username):
        raise ValueError("ユーザー名は小文字英数字と . _ - (2〜32 文字) にしてください")
    if len(password) < 8:
        raise ValueError("パスワードは 8 文字以上にしてください")
    users = load_users(state_dir)
    salt = secrets.token_bytes(16)
    users[username] = {"salt": salt.hex(), "hash": _hash(password, salt),
                       "name": display_name or users.get(username, {}).get("name") or username}
    _save(state_dir, users)


def remove_user(state_dir: Path, username: str) -> bool:
    users = load_users(state_dir)
    if users.pop(username, None) is None:
        return False
    _save(state_dir, users)
    return True


def verify(state_dir: Path, username: str, password: str) -> bool:
    u = load_users(state_dir).get(username)
    if not u:
        _hash(password, b"0" * 16)  # 存在しないユーザーでも同程度の時間をかける
        return False
    return hmac.compare_digest(u["hash"], _hash(password, bytes.fromhex(u["salt"])))


def secret_key(state_dir: Path) -> str:
    """セッション署名用の鍵。初回に生成して .state/ に保存する。"""
    path = state_dir / "web_secret"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        os.chmod(path, 0o600)
    return path.read_text().strip()
