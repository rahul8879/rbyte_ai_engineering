import time
import logging
from functools import wraps
import requests # We'll use this for simulating API calls


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
api_logger = logging.getLogger('API_Monitor')

# --- The Decorator for API Call Logging ---
def log_api_call(api_name):
    """
    A decorator to log the duration and outcome of an API call.

    Args:
        api_name (str): The name of the API being called (e.g., 'WeatherAPI', 'PaymentGateway').
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            api_logger.info(f"Initiating call to {api_name}...")
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = (end_time - start_time) * 1000 # in milliseconds
                api_logger.info(f"Successfully called {api_name}. Duration: {duration:.2f} ms")
                return result
            except Exception as e:
                end_time = time.time()
                duration = (end_time - start_time) * 1000 # in milliseconds
                api_logger.error(f"Failed to call {api_name}. Error: {e}. Duration: {duration:.2f} ms")
                # Re-raise the exception so the calling code can handle it if needed
                raise
        return wrapper
    return decorator

# --- Example Usage: Simulating API Calls ---

# Simulate a successful API call
@log_api_call(api_name="WeatherAPI")
def get_weather_data(city):
    """Simulates fetching weather data from an external API."""
    api_logger.info(f"Fetching weather for {city}...")
    # Simulate network delay and processing
    time.sleep(1.5)
    # Simulate a successful response
    return {"city": city, "temperature": "25C", "condition": "Sunny"}

# Simulate a failing API call (e.g., due to network error or invalid request)
@log_api_call(api_name="PaymentGatewayAPI")
def process_payment(amount, currency):
    """Simulates processing a payment through an external gateway."""
    api_logger.info(f"Processing payment of {amount} {currency}...")
    # Simulate a delay
    time.sleep(0.8)
    # Simulate a random failure
    if amount > 1000:
        api_logger.error("Payment amount too high, simulating failure.")
        raise ValueError("Payment denied: Amount exceeds limit")
    else:
        # Simulate a successful response
        return {"transaction_id": "TXN12345", "status": "approved"}

# Simulate an API call that might fail due to external factors (e.g., connection error)
@log_api_call(api_name="StockMarketAPI")
def get_stock_price(symbol):
    """Simulates fetching stock price using requests, might fail due to connectivity."""
    try:
        api_logger.info(f"Fetching price for {symbol}...")
        # This will actually try to hit a URL.
        # Use a real URL or a placeholder like 'http://example.com'
        # For a guaranteed failure without internet, use a non-existent domain or port
        response = requests.get(f"https://www.google.com/search?q={symbol}_stock_price", timeout=2)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return {"symbol": symbol, "price": "150.75", "currency": "USD"}
    except requests.exceptions.RequestException as e:
        api_logger.error(f"Request error for {symbol}: {e}")
        raise # Re-raise the original exception

# --- Running the examples ---
if __name__ == "__main__":
    print("\n--- Calling WeatherAPI ---")
    try:
        weather_data = get_weather_data("London")
        print(f"Weather Data: {weather_data}")
    except Exception as e:
        print(f"Handled error in main for WeatherAPI: {e}")

    print("\n--- Calling PaymentGatewayAPI (Successful) ---")
    try:
        payment_status = process_payment(500, "USD")
        print(f"Payment Status: {payment_status}")
    except Exception as e:
        print(f"Handled error in main for PaymentGatewayAPI: {e}")

    print("\n--- Calling PaymentGatewayAPI (Failing) ---")
    try:
        payment_status = process_payment(1500, "EUR")
        print(f"Payment Status: {payment_status}")
    except Exception as e:
        print(f"Handled error in main for PaymentGatewayAPI: {e}") # This will catch the ValueError

    print("\n--- Calling StockMarketAPI (Potentially Failing due to network) ---")
    try:
        stock_price = get_stock_price("MSFT")
        print(f"Stock Price: {stock_price}")
    except Exception as e:
        print(f"Handled error in main for StockMarketAPI: {e}") # This will catch the requests.exceptions.RequestException

    print("\n--- Calling StockMarketAPI (Another Symbol) ---")
    try:
        stock_price = get_stock_price("GOOG")
        print(f"Stock Price: {stock_price}")
    except Exception as e:
        print(f"Handled error in main for StockMarketAPI: {e}")