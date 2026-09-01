"""Manual, side-effecting smoke test against the real ipinfo.io API.

Not part of the pytest suite (test_module_c.py mocks ipinfo.io via FakeIpinfoSession throughout —
see its docstring) — running this script makes one real HTTP call. Not run automatically for that
reason; run it yourself when ready:

    uv run python -m module_c.manual_ipinfo_check

Requires a real IPINFO_TOKEN (free signup at ipinfo.io, 50k req/month) in the environment or a
.env file at the repo root — the placeholder value shipped in .env.example won't work.
"""

from __future__ import annotations

from dotenv import load_dotenv

from module_c import ipinfo_client


def main() -> None:
    load_dotenv()

    print("Looking up this machine's public-IP location via ipinfo.io...")
    location = ipinfo_client.locate_ip()

    print(f"ip={location.ip}")
    print(f"city={location.city}")
    print(f"region={location.region}")
    print(f"country={location.country}")
    print(f"lat={location.lat}")
    print(f"lon={location.lon}")


if __name__ == "__main__":
    main()
