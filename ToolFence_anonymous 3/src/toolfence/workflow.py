"""Controller-owned evidence for narrow, necessary shopping dependencies.

Tool output is data, never an instruction.  These predicates use typed product
records and a checkout-specific OTP envelope; arbitrary text cannot expand the
agent's authority.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from typing import Any


_PURCHASE = re.compile(r"\b(buy|purchase|order)\b", re.IGNORECASE)
_MULTIPLE = re.compile(r"\b(?:[2-9]|\d{2,}|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)
_OTP = re.compile(r"\b(?:checkout\s+)?OTP\s+is:\s*(\d{6})\b", re.IGNORECASE)


class CheckoutPhase(str, Enum):
    """Controller-owned states; tool text can only supply typed transition evidence."""

    COLLECTING = "collecting"
    PRODUCT_FOUND = "product_found"
    CART_READY = "cart_ready"
    AWAITING_OTP = "awaiting_otp"
    OTP_READY = "otp_ready"
    COMPLETED = "completed"


def checkout_otp_messages(value: Any) -> list[dict[str, str]]:
    """Extract only typed checkout OTP data from linked inbox records."""
    if not isinstance(value, list):
        return []
    result = []
    for email in value:
        if hasattr(email, "model_dump"):
            email = email.model_dump()
        if not isinstance(email, Mapping):
            continue
        if (str(email.get("sender", "")).casefold() != "shopping_platform@mail.com"
                or str(email.get("subject", "")).casefold() != "shopping platform checkout otp"):
            continue
        match = _OTP.search(str(email.get("body", "")))
        if match:
            result.append({
                "sender": "shopping_platform@mail.com",
                "subject": "Shopping Platform Checkout OTP",
                "body": f"Your shopping platform checkout OTP is: {match.group(1)}",
            })
    return result


class WorkflowEvidence:
    def __init__(self, query: str):
        self.query = query
        self.completed: list[str] = []
        self.product_ids: set[str] = set()
        self.cart_ids: set[str] = set()
        self.discount_codes: set[str] = set()
        self.phase = CheckoutPhase.COLLECTING
        self._checkout_otp: str | None = None
        self.checkout_ids: frozenset[str] = frozenset()

    @property
    def pending_checkout(self) -> bool:
        return self.phase in {CheckoutPhase.AWAITING_OTP, CheckoutPhase.OTP_READY}

    @pending_checkout.setter
    def pending_checkout(self, value: bool) -> None:
        # Keep the legacy test/setup interface while making phase authoritative.
        if value:
            self.phase = (
                CheckoutPhase.OTP_READY if self._checkout_otp else CheckoutPhase.AWAITING_OTP
            )
        elif self.pending_checkout:
            self.phase = CheckoutPhase.COMPLETED
            self._checkout_otp = None

    @property
    def checkout_otp(self) -> str | None:
        return self._checkout_otp

    @checkout_otp.setter
    def checkout_otp(self, value: str | None) -> None:
        self._checkout_otp = value
        if self.pending_checkout:
            self.phase = CheckoutPhase.OTP_READY if value else CheckoutPhase.AWAITING_OTP

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "completed_tools": self.completed[-8:],
            "matching_product_ids": sorted(self.product_ids)[:20],
            "cart_product_ids": sorted(self.cart_ids)[:20],
            "available_discount_codes": sorted(self.discount_codes)[:20],
            "checkout_product_ids": sorted(self.checkout_ids)[:20],
            "checkout_verification_pending": self.pending_checkout,
            "checkout_otp_observed": self.checkout_otp is not None,
        }

    def observe(self, tool: str, arguments: Mapping[str, Any], value: Any) -> None:
        self.completed.append(tool)
        if len(self.completed) > 32:
            self.completed = self.completed[-32:]
        if tool == "search_product" and isinstance(value, Mapping):
            search_term = str(arguments.get("product_name") or "").strip()
            if not (_PURCHASE.search(self.query) and search_term
                    and search_term.casefold() in self.query.casefold()):
                return
            for title, record in value.items():
                if not isinstance(record, Mapping) or not isinstance(title, str):
                    continue
                product_id = record.get("product_id")
                if (search_term.casefold() in title.casefold()
                        and isinstance(product_id, str) and product_id):
                    self.product_ids.add(product_id)
            if self.product_ids and self.phase == CheckoutPhase.COLLECTING:
                self.phase = CheckoutPhase.PRODUCT_FOUND
        elif tool == "cart_add_product" and isinstance(value, str):
            product_id = arguments.get("product_id")
            if (self.phase in {CheckoutPhase.COLLECTING, CheckoutPhase.PRODUCT_FOUND,
                               CheckoutPhase.CART_READY}
                    and isinstance(product_id, str)
                    and (product_id in self.product_ids
                         or product_id.casefold() in self.query.casefold())
                    and "added to cart" in value.casefold()):
                self.cart_ids.add(product_id)
                self.phase = CheckoutPhase.CART_READY
        elif tool == "view_available_discounts" and isinstance(value, str):
            self.discount_codes = set(re.findall(
                r"^\s*-\s*Code:\s*([^,\s]+)\s*,", value, re.MULTILINE,
            ))
        elif tool == "checkout_selected_cart" and isinstance(value, str):
            selected = arguments.get("product_ids")
            if (self.phase == CheckoutPhase.CART_READY
                    and isinstance(selected, list) and selected
                    and all(isinstance(item, str) for item in selected)
                    and len(selected) == len(set(selected))
                    and set(selected) == self.cart_ids
                    and "please verify your account to proceed the checkout" in value.casefold()):
                self.checkout_ids = frozenset(selected)
                self._checkout_otp = None
                self.phase = CheckoutPhase.AWAITING_OTP
        elif tool in {"get_shopping_linked_received_emails", "get_shopping_linked_unread_emails"}:
            if self.phase not in {CheckoutPhase.AWAITING_OTP, CheckoutPhase.OTP_READY}:
                return
            messages = checkout_otp_messages(value)
            codes = {message["body"].rsplit(" ", 1)[-1] for message in messages}
            # Conflicting codes invalidate the token instead of choosing one.
            self._checkout_otp = next(iter(codes)) if len(codes) == 1 else None
            self.phase = CheckoutPhase.OTP_READY if self._checkout_otp else CheckoutPhase.AWAITING_OTP
        elif tool in {"verify_shopping_account", "verify_transaction"}:
            if (self.phase == CheckoutPhase.OTP_READY and isinstance(value, str)
                    and value.startswith("Account verified.")):
                self._checkout_otp = None
                self.phase = CheckoutPhase.COMPLETED

    def decide(self, tool: str, arguments: Mapping[str, Any]) -> tuple[str, str, str] | None:
        """Return (status, reason, repair step) only for recognized workflow edges."""
        if not _PURCHASE.search(self.query):
            return None
        if tool == "cart_add_product":
            if self.phase in {CheckoutPhase.AWAITING_OTP, CheckoutPhase.OTP_READY,
                              CheckoutPhase.COMPLETED}:
                return ("deny", "Cart cannot change after checkout has begun", "")
            product_id = arguments.get("product_id")
            quantity = arguments.get("quantity")
            if not set(arguments).issubset({"product_id", "quantity"}):
                return ("deny", "Cart call has undeclared parameters", "")
            if isinstance(product_id, str) and product_id in self.cart_ids:
                return ("deny", "Product is already in the cart", "")
            if isinstance(product_id, str) and product_id.casefold() in self.query.casefold():
                return None  # user-specified identifier is handled by the blueprint/judge
            if not isinstance(product_id, str) or product_id not in self.product_ids:
                return ("needs_evidence", "Product ID has not been tied to the requested product",
                        "Search for the requested product and use its product_id.")
            if _MULTIPLE.search(self.query):
                return None  # a stated quantity needs semantic review
            if quantity != 1:
                return ("deny", "Quantity differs from the single requested product", "")
            return ("allow", "One requested product from the matching search result", "")
        if tool == "checkout_selected_cart":
            if self.phase in {CheckoutPhase.AWAITING_OTP, CheckoutPhase.OTP_READY,
                              CheckoutPhase.COMPLETED}:
                return ("deny", "Checkout has already begun for this request", "")
            if not self.cart_ids:
                return ("needs_evidence", "No requested product has been added to the cart",
                        "Add the requested product to the cart before checkout.")
            if not set(arguments).issubset({"product_ids", "discount_code"}):
                return ("deny", "Checkout call has undeclared parameters", "")
            selected = arguments.get("product_ids")
            if (not isinstance(selected, list) or not selected
                    or any(not isinstance(item, str) for item in selected)
                    or len(selected) != len(set(selected))
                    or set(selected) != self.cart_ids):
                return ("deny", "Checkout selection is not the verified cart selection", "")
            discount = arguments.get("discount_code")
            if discount is not None:
                if not isinstance(discount, str) or not discount:
                    return ("deny", "Checkout discount code is malformed", "")
                if discount not in self.discount_codes:
                    return ("needs_evidence", "Discount code was not observed in available discounts",
                            "Read the available discounts before checkout.")
            return ("allow", "Checkout selection matches products added to the cart", "")
        if tool in {"verify_shopping_account", "verify_transaction"}:
            if set(arguments) != {"otp"}:
                return ("deny", "Verification call has undeclared parameters", "")
            if self.phase not in {CheckoutPhase.AWAITING_OTP, CheckoutPhase.OTP_READY}:
                return ("needs_evidence", "No requested checkout is awaiting verification",
                        "Checkout the requested items before attempting verification.")
            if self.phase != CheckoutPhase.OTP_READY or self.checkout_otp is None:
                return ("needs_evidence", "Checkout OTP has not been observed in the linked inbox",
                        "Read the shopping linked inbox for the checkout OTP.")
            if arguments.get("otp") != self.checkout_otp:
                return ("deny", "OTP does not match the pending checkout message", "")
            return ("allow", "OTP matches the pending checkout message", "")
        return None
