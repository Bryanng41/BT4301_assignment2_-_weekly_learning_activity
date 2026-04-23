"""Reserved for shared encoding helpers; purchase API uses bundle + purchase_prediction only."""


def parse_customer_id(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
