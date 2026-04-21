"""HTTP client to call cloud POST /v1/pii/mask for PII masking."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("memra.pii_client")


class PiiClient:
    """Calls the Memra cloud PII masking endpoint to mask content before sync."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    def mask_batch(
        self, contents: list[str], tenant_id: str = "default"
    ) -> list[str] | None:
        """Mask PII in a batch of content strings via cloud endpoint.

        Args:
            contents: List of content strings to mask.
            tenant_id: Tenant scope for PII masking.

        Returns:
            List of masked content strings on success, None on any failure.
        """
        try:
            response = httpx.post(
                f"{self._api_url}/pii/mask",
                json={"contents": contents, "tenant_id": tenant_id},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return [r["masked_content"] for r in data["results"]]
        except Exception as exc:
            logger.warning("PII masking request failed: %s", exc)
            return None
