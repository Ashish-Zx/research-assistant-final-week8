# agent/tools/currency.py
import requests


def convert_currency(amount: str, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using current exchange rates."""
    try:
        amount_float = float(amount)  # accepts both numeric and string inputs from the LLM
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        rates = data.get("rates", {})
        rate = rates.get(to_currency.upper())
        if rate is None:
            return f"Currency '{to_currency}' not supported."
        converted = amount_float * rate
        return f"{amount_float:.2f} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"
    except Exception as e:
        return f"Currency conversion error: {e}"
