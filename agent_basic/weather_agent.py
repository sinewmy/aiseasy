import re
from typing import Optional

import requests


class WeatherAgent:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def handle_user_request(self, user_request: str) -> str:
        city = self._extract_city(user_request)
        if not city:
            return "Please specify a location, for example: 'What is the weather in London?'"

        return self.get_current_weather(city)

    def _extract_city(self, text: str) -> Optional[str]:
        text = text.strip().lower()
        patterns = [
            r"weather in (?P<city>.+)",
            r"weather for (?P<city>.+)",
            r"current weather in (?P<city>.+)",
            r"check weather in (?P<city>.+)",
            r"in (?P<city>.+)",
            r"for (?P<city>.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                city = match.group("city").strip(" ?.")
                return city

        if "weather" in text:
            text = re.sub(r"^(what|whats|what's|show|tell|check)\b", "", text)
            text = text.replace("weather", "").strip(" ?.")
            if text:
                return text

        return None

    def get_current_weather(self, city: str) -> str:
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",
        }
        response = requests.get(self.base_url, params=params, timeout=10)
        if response.status_code != 200:
            return self._format_error(response)

        data = response.json()
        weather = data.get("weather", [{}])[0]
        main = data.get("main", {})
        wind = data.get("wind", {})

        condition = weather.get("description", "Unknown").capitalize()
        temperature = main.get("temp")
        humidity = main.get("humidity")
        wind_speed = wind.get("speed")

        return (
            f"Current weather in {data.get('name', city)}: {condition}. "
            f"Temperature: {temperature}°C, Humidity: {humidity}%, Wind speed: {wind_speed} m/s."
        )

    def _format_error(self, response: requests.Response) -> str:
        try:
            data = response.json()
            message = data.get("message", "Unable to fetch weather data.")
        except ValueError:
            message = "Unable to fetch weather data."
        return f"Error: {message.capitalize()}"
