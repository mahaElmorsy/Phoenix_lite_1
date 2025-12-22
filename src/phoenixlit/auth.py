"""phoenixlit.auth

local PIN protection.

- In its First run : creates data/auth/pin.json ===> Increase PIN Encryption (salt + PBKDF2 hash) .
- Every run asks for the PIN ===> (hidden input to add more protection).
- Failed attempts are counted in data/auth/failed.txt ====> will help in (future work : add it later for warning the victim + upgrade security level in the future if a lot of attempts detected) .
"""

import json
from dataclasses import dataclass
from getpass import getpass
from hashlib import pbkdf2_hmac
from pathlib import Path
import os


@dataclass(frozen=True)
class PinConfig:
    pin_file: Path
    failed_file: Path
    max_attempts: int = 3 #limits failed attempts to 3


def _hash_pin(pin: str, salt_hex: str, iterations: int) -> str:
    salt = bytes.fromhex(salt_hex)
    derived_key = pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations)
    return derived_key.hex()


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_pin(pin_file: Path) -> None:
    """check pin.json and create it if it doesn't exist."""
    if pin_file.exists():
        return

    print("[Phoenix] PIN file not found. Let's create one now.")
    while True:
        pin1 = getpass("Create a PIN (numbers only recommended): ").strip()
        pin2 = getpass("Confirm PIN: ").strip()
        if not pin1:
            print("[Phoenix] PIN cannot be empty.")
            continue
        if pin1 != pin2:
            print("[Phoenix] PINs do not match. Try again.")
            continue
        break

    salt_hex = os.urandom(16).hex()
    iterations = 150_000
    rec = {
        "salt_hex": salt_hex,
        "iterations": iterations,
        "hash_hex": _hash_pin(pin1, salt_hex, iterations),
    }
    _write_json(pin_file, rec)
    print(f"[Phoenix] PIN saved to: {pin_file}")


def _read_failed_count(failed_file: Path) -> int:
    if not failed_file.exists():
        return 0
    try:
        return int(failed_file.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        return 0


def _write_failed_count(failed_file: Path, failed_count: int) -> None:
    failed_file.parent.mkdir(parents=True, exist_ok=True)
    failed_file.write_text(str(int(failed_count)), encoding="utf-8")


def verify_pin(auth_config: PinConfig):
    """
        Ask user for PIN.
        Returns a warning string (or None).
        Close if max attempts exceeded.
    """
    ensure_pin(auth_config.pin_file)

    rec = _load_json(auth_config.pin_file) or {}
    salt_hex = rec.get("salt_hex")
    hash_hex = rec.get("hash_hex")
    iterations = int(rec.get("iterations", 150_000))

    if not salt_hex or not hash_hex:
        return "PIN record is invalid. Delete pin.json and re-create it."

    failed = _read_failed_count(auth_config.failed_file)
    if failed >= auth_config.max_attempts:
        raise SystemExit("[Phoenix] Too many failed PIN attempts. Access denied.")

    pin = getpass("Enter PIN: ").strip()
    is_pin_correct = _hash_pin(pin, salt_hex, iterations) == hash_hex

    if is_pin_correct:
        # reset
        _write_failed_count(auth_config.failed_file, 0)
        return None

    failed += 1
    _write_failed_count(auth_config.failed_file, failed)

    remaining = max(auth_config.max_attempts - failed, 0)
    if remaining <= 0:
        raise SystemExit("[Phoenix] Too many failed PIN attempts. Access denied.")
    return f"Wrong PIN. Remaining attempts: {remaining}"
