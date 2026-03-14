"""Server location detection via IP geolocation."""

import requests

ELECTRICITY_PRICES = {
    "US": ("USD", "USD ", 0.12),
    "CA": ("CAD", "CAD ", 0.10),
    "GB": ("GBP", "GBP ", 0.28),
    "DE": ("EUR", "EUR ", 0.30),
    "FR": ("EUR", "EUR ", 0.18),
    "NL": ("EUR", "EUR ", 0.22),
    "SE": ("EUR", "EUR ", 0.12),
    "NO": ("EUR", "EUR ", 0.10),
    "TW": ("TWD", "TWD ", 2.53),
    "SG": ("SGD", "SGD ", 0.18),
    "JP": ("JPY", "JPY ", 0.22),
    "KR": ("KRW", "KRW ", 0.11),
    "AU": ("AUD", "AUD ", 0.25),
    "default": ("USD", "USD ", 0.12),
}

def get_server_location() -> dict:
    """
    Returns:
        {
            "ip": "1.2.3.4",
            "country_code": "US",
            "country": "United States",
            "region": "California",
            "city": "San Jose",
            "currency_code": "USD",
            "currency_symbol": "USD ",
            "electricity_price_kwh": 0.12,
            "source": "ip-api.com" | "default"
        }
    """
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                country_code = data.get("countryCode", "default")
                currency_code, currency_symbol, price = ELECTRICITY_PRICES.get(
                    country_code, ELECTRICITY_PRICES["default"]
                )
                return {
                    "ip": data.get("query", "unknown"),
                    "country_code": country_code,
                    "country": data.get("country", "Unknown"),
                    "region": data.get("regionName", "Unknown"),
                    "city": data.get("city", "Unknown"),
                    "currency_code": currency_code,
                    "currency_symbol": currency_symbol,
                    "electricity_price_kwh": price,
                    "source": "ip-api.com",
                }
    except Exception:
        pass

    return {
        "ip": "unknown",
        "country_code": "default",
        "country": "Unknown",
        "region": "Unknown",
        "city": "Unknown",
        "currency_code": "USD",
        "currency_symbol": "USD ",
        "electricity_price_kwh": 0.12,
        "source": "default",
    }


if __name__ == "__main__":
    loc = get_server_location()
    print(f"Server location: {loc['city']}, {loc['region']}, {loc['country']}")
    print(f"Electricity price: {loc['currency_symbol']}{loc['electricity_price_kwh']}/kWh")
    print(f"Source: {loc['source']}")
