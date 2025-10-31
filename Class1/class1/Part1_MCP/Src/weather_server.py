from fastmcp import FastMCP

server = FastMCP(name="Weather Server")

@server.tool()
def get_weather(city: str):
    """Return fake weather info for demonstration."""
    return f"The weather in {city} is sunny and 25°C."

if __name__ == "__main__":
    server.run()
