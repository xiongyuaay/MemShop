from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from retail_thor.catalog import match_product


class WorldKnowledgeManager:
    """Product knowledge API used by NPC-side planners."""

    def __init__(self, catalog: Iterable[dict[str, Any]]) -> None:
        self._products = [deepcopy(product) for product in catalog]
        self._by_id = {product["product_id"]: product for product in self._products if product.get("product_id")}
        self._call_log: list[dict[str, Any]] = []

    def get_product(self, product_id: str) -> dict[str, Any]:
        self._record("get_product", {"product_id": product_id})
        product = self._by_id.get(product_id)
        if product is None:
            raise KeyError(f"unknown product_id: {product_id}")
        return deepcopy(product)

    def search_products(self, constraints: dict[str, Any] | None = None, text_query: str | None = None) -> list[dict[str, Any]]:
        payload = {"constraints": constraints or {}, "text_query": text_query or ""}
        self._record("search_products", payload)
        results = []
        for product in self._products:
            if constraints and not match_product(product, constraints):
                continue
            if text_query and not _matches_text(product, text_query):
                continue
            results.append(deepcopy(product))
        return results

    def get_candidate_attributes(self, product_ids: Iterable[str]) -> list[dict[str, Any]]:
        product_id_list = list(product_ids)
        self._record("get_candidate_attributes", {"product_ids": product_id_list})
        summaries = []
        for product_id in product_id_list:
            product = self.get_product(product_id)
            summaries.append(_product_summary(product))
        return summaries

    def consume_call_log(self) -> list[dict[str, Any]]:
        calls = list(self._call_log)
        self._call_log.clear()
        return calls

    def _record(self, api: str, args: dict[str, Any]) -> None:
        self._call_log.append({"api": api, "args": deepcopy(args)})


def _matches_text(product: dict[str, Any], text_query: str) -> bool:
    query = text_query.casefold()
    fields = [
        product.get("display_name_en", ""),
        product.get("display_name_zh", ""),
        product.get("category", ""),
        product.get("brand", ""),
        *product.get("attributes", []),
        *product.get("synonyms_zh", []),
    ]
    return any(str(field).casefold() in query or query in str(field).casefold() for field in fields if field)


def _product_summary(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": product.get("product_id"),
        "display_name_en": product.get("display_name_en", ""),
        "display_name_zh": product.get("display_name_zh", ""),
        "category": product.get("category", ""),
        "attributes": list(product.get("attributes", [])),
        "brand": product.get("brand", ""),
        "shelf_id": product.get("shelf_id", ""),
    }
