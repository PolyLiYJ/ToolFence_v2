"""Narrow, source-bound authorization for first-party workflow prerequisites."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_ACCOUNT_TASK = re.compile(
    r"\b(buy|purchase|order|checkout|cart|shop|account|log\s?in)\b|购买|下单|购物|登录|账户",
    re.IGNORECASE,
)
_PURCHASE_TASK = re.compile(r"\b(buy|purchase|checkout|order|rebuy)\b|购买|下单|结账|再买", re.IGNORECASE)


def _matches_requested_product(query: str, product_name: str) -> bool:
    query_words = set(re.findall(r"[a-z0-9]+", query.casefold()))
    product_words = re.findall(r"[a-z0-9]+", product_name.casefold())
    distinctive = [word for word in product_words if len(word) >= 4]
    return bool(distinctive) and (
        product_name.casefold() in query.casefold()
        or (len(distinctive) >= 2 and len(set(distinctive) & query_words) >= 2)
    )


def _checkout_matches_cart(
    query: str, arguments: Mapping[str, Any], observed: Mapping[str, Any],
) -> bool:
    if not _PURCHASE_TASK.search(query):
        return False
    if not set(arguments).issubset({"product_ids", "discount_code"}):
        return False
    product_ids = arguments.get("product_ids")
    if not isinstance(product_ids, list) or not product_ids:
        return False
    if not all(isinstance(item, str) and item for item in product_ids):
        return False
    if len(set(product_ids)) != len(product_ids):
        return False
    cart = observed.get("view_cart")
    if not isinstance(cart, Mapping):
        return False
    cart_items = {
        item.get("product_id"): item
        for item in cart.values() if isinstance(item, Mapping)
    }
    searches = observed.get("search_product")
    if not isinstance(searches, list):
        return False
    for product_id in product_ids:
        item = cart_items.get(product_id)
        if item is None or not _matches_requested_product(
            query, str(item.get("product_name", "")),
        ):
            return False
        if not any(
            isinstance(result, Mapping)
            and any(
                isinstance(info, Mapping)
                and info.get("product_id") == product_id
                and str(name).casefold() == str(item.get("product_name", "")).casefold()
                for name, info in result.items()
            )
            for result in searches
        ):
            return False
        quantity = item.get("quantity")
        quantity_words = {2: "two", 3: "three", 4: "four", 5: "five"}
        if not isinstance(quantity, int) or quantity < 1:
            return False
        word = quantity_words.get(quantity)
        if quantity != 1 and not (
            re.search(rf"\b{quantity}\b", query)
            or (word is not None and re.search(rf"\b{word}\b", query, re.IGNORECASE))
        ):
            return False
    discount = arguments.get("discount_code")
    if discount is not None:
        if not isinstance(discount, str) or not discount:
            return False
        available = observed.get("view_available_discounts")
        if not isinstance(available, str):
            return False
        codes = re.findall(r"^\s*-\s*Code:\s*([^,\s]+)\s*,", available, re.MULTILINE)
        if discount not in codes:
            return False
    return True


def verified_workflow_prerequisite(
    query: str,
    function: str,
    arguments: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> bool:
    """Allow narrow shopping steps supported by independent structured reads.

    Login needs exact credentials from the account tool. Checkout needs a cart
    entry, a matching product search result, and an account discount when one
    is supplied. A value merely appearing in an email or product description
    cannot satisfy these checks. Other payments, messages, and account changes
    remain outside this rule.
    """
    if function == "checkout_selected_cart":
        return _checkout_matches_cart(query, arguments, observed)
    if function != "login_shopping_account" or not _ACCOUNT_TASK.search(query):
        return False
    if set(arguments) != {"username", "password"}:
        return False
    account = observed.get("get_shopping_account_user_information")
    if not isinstance(account, Mapping):
        return False
    fields = {str(key).casefold(): value for key, value in account.items()}
    username, password = fields.get("username"), fields.get("password")
    return bool(
        isinstance(username, str) and username
        and isinstance(password, str) and password
        and arguments["username"] == username
        and arguments["password"] == password
    )


def verified_discount_web_step(
    query: str,
    function: str,
    arguments: Mapping[str, Any],
    observed: Mapping[str, Any],
    env: Any,
) -> bool:
    """Authorize a discount form by its controller-owned service effect.

    Email text only proposes a URL. The environment's service registry must
    independently show that the exact page can only add a discount to the
    current account and accepts only its email address. A page that changes a
    password, downloads a file, or accepts another field fails this check.
    """
    lowered = query.casefold()
    if "email" not in lowered or "discount" not in lowered:
        return False
    if function not in {"browse_webpage", "input_to_webpage"}:
        return False
    url = arguments.get("url")
    if not isinstance(url, str) or not url:
        return False
    normalized = re.sub(r"^https?://", "", url.casefold())
    emails = observed.get("get_shopping_linked_unread_emails")
    if not isinstance(emails, list):
        return False
    if not any(
        isinstance(getattr(email, "body", None), str)
        and normalized in email.body.casefold()
        for email in emails
    ):
        return False
    database = getattr(env, "web_database", None)
    pages = getattr(database, "web_list", None)
    if not isinstance(pages, list):
        return False
    page = next(
        (item for item in pages if str(getattr(item, "web_url", "")).casefold() == normalized),
        None,
    )
    if page is None:
        return False
    handlers = getattr(page, "web_function", None)
    if not isinstance(handlers, list) or len(handlers) != 1:
        return False
    handler = handlers[0]
    if getattr(handler, "handler", None) != "add_discount":
        return False
    if getattr(page, "field_to_parameters", None) != {"email_field": "email"}:
        return False
    static = getattr(handler, "static_parameters", None)
    if not isinstance(static, Mapping) or set(static) != {"code", "percentage"}:
        return False
    if not isinstance(static["code"], str) or not static["code"]:
        return False
    if not isinstance(static["percentage"], (int, float)) or not 0 < static["percentage"] <= 1:
        return False
    if function == "browse_webpage":
        return set(arguments) == {"url"}
    account = observed.get("get_shopping_account_user_information")
    if not isinstance(account, Mapping):
        return False
    email = account.get("Email")
    return bool(
        isinstance(email, str) and email
        and set(arguments) == {"url", "input_ids", "input_values"}
        and arguments["input_ids"] == ["email_field"]
        and arguments["input_values"] == [email]
    )


def verified_password_otp_step(
    query: str,
    function: str,
    arguments: Mapping[str, Any],
    env: Any,
    decisions: Sequence[Mapping[str, Any]],
    completed_tools: Sequence[str],
) -> bool:
    """Finish an authorized password change using its pending OTP challenge."""
    if function != "verify_shopping_account" or set(arguments) != {"otp"}:
        return False
    otp = arguments.get("otp")
    if not isinstance(otp, str) or len(otp) < 4:
        return False
    if not ("password" in query.casefold() or "密码" in query):
        return False
    if "get_shopping_linked_unread_emails" not in completed_tools:
        return False
    platform = getattr(env, "shopping_platform", None)
    username = getattr(platform, "current_account_username", None)
    accounts = getattr(platform, "account_list", None)
    if not isinstance(accounts, list):
        return False
    account = next(
        (item for item in accounts if getattr(item, "username", None) == username),
        None,
    )
    if account is None:
        return False
    stack = getattr(account, "verification_stack", None)
    if not isinstance(stack, Mapping):
        return False
    pending = stack.get(otp)
    if not isinstance(pending, tuple) or len(pending) != 2:
        return False
    operation, values = pending
    if getattr(operation, "__name__", None) != "update_shopping_account_password":
        return False
    if not isinstance(values, tuple) or not values:
        return False
    new_password = values[-1]
    if not isinstance(new_password, str) or len(new_password) < 4:
        return False
    if new_password.casefold() not in query.casefold():
        return False
    return any(
        record.get("tool") == "update_shopping_account_password"
        and record.get("allowed") is True
        and isinstance(record.get("arguments"), Mapping)
        and record["arguments"].get("password") == new_password
        for record in decisions
    )
