#!/usr/bin/env python3
"""
Diagnostic script to check Langfuse environment and client initialization.
Run this locally and paste the full output here (redact secrets) so I can inspect the exact error.
"""
import os
import traceback

try:
    from langfuse import Langfuse
except Exception as exc:
    print("Failed to import langfuse:", repr(exc))
    raise


def _mask(val: str | None) -> str | None:
    if val is None:
        return None
    v = val.strip()
    return f"{v[:8]}... (len={len(v)})"


keys = [
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_API_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
    "ENV",
]

print("--- Langfuse environment diagnostics ---")
for k in keys:
    print(f"{k}: {repr(os.getenv(k))} -> masked: {_mask(os.getenv(k))}")

public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or os.getenv("LANGFUSE_API_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
env = os.getenv("ENV") or "prod"

print("\nComputed init kwargs:")
print("  public_key ->", _mask(public_key))
print("  secret_key ->", _mask(secret_key))
print("  host       ->", repr(host))
print("  environment->", env)

print("\nAttempting to instantiate Langfuse client...")
kwargs = {}
if public_key:
    kwargs["public_key"] = public_key
if secret_key:
    kwargs["secret_key"] = secret_key
if host:
    kwargs["host"] = host
kwargs["environment"] = env

try:
    client = Langfuse(**kwargs)
    print("Langfuse client created:", repr(client))
    # Do not call network ops here; instantiation may already validate keys.
except Exception as exc:
    print("Exception during Langfuse initialization:")
    print(repr(exc))
    traceback.print_exc()
else:
    print(
        "\nAttempting to start a lightweight span (this will hit the Langfuse server)."
    )
    try:
        span = client.start_span(name="diagnostic.span", input={"test": True})
        print("Started span:", getattr(span, "id", None))
        span.update(output={"ok": True})
        span.end()
        print("Span created and ended successfully.")
        try:
            client.flush()
            print("Flush succeeded.")
        except Exception as exc:
            print("Flush failed:", repr(exc))
            traceback.print_exc()
    except Exception as exc:
        print("Exception while starting/using Langfuse client (network/auth):")
        print(repr(exc))
        traceback.print_exc()

print("--- end diagnostics ---")
