Check the Src and Output folders.
Src: the source codes for customized MCP servers and claude configuration file
Output: the screenshots for launching MCP servers by parsing input from claude.

Claude does not support @mcp.prompt() and only partial support @mcp.resource()

Find out the following informations during integrating customized MCP servers with Claude.

@mcp.tool() → ✅ visible and callable

@mcp.resource() → ⚙️ visible but not callable

@mcp.prompt() → ❌ not recognized yet

| Decorator         | Works in Claude Desktop? | Use Case                          |
| ----------------- | ------------------------ | --------------------------------- |
| `@mcp.tool()`     | ✅ Yes                    | Executable functions              |
| `@mcp.resource()` | ⚙️ Partial               | Static data / read-only           |
| `@mcp.prompt()`   | ❌ Not yet                | Prompt templates (future support) |
