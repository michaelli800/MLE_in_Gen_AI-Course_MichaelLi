from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test Server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def greeting(name: str) -> str:
    """Return a personalized greeting."""
    return f"Hello, {name}!"
    
@mcp.tool()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting message."""
    return f"Write a {style} greeting for someone named {name}."
    
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


@mcp.prompt()
def greeting_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    return f"Write a {style} greeting for someone named {name}."