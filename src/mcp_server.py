"""
MCP server — VITA's tool layer published over stdio.

    python -m src.mcp_server

A protocol wrapper and nothing more. Every tool is implemented once, in
`src.tools`, and both consumers execute the same code: the running application
uses the tool layer directly, and an MCP client reaches it through here. They
cannot drift apart, because there is only one implementation to drift from.

What an external client gets is the **full** surface, decision tools included. A
person or agent driving VITA deliberately through MCP is in a different position
from the conversation model, which is reading text typed by a member of the
public: the conversation model is advertised the retrieval tier only, and has no
tool that assigns an urgency. The tier split lives in `src.tools`; this module
just exposes what it is given.

Startup is deliberately shared with the web application, so an MCP client sees
the same knowledge base, the same rules and the same case store that a patient
using the web interface would.

Note on stdout: MCP speaks JSON-RPC over it, so nothing else may be written
there. All diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from dotenv import load_dotenv

from .config import APP_NAME, APP_VERSION
from .services.container import VitaServices
from .tools import ToolLayer

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(levelname)-7s %(name)s  %(message)s",
)
logger = logging.getLogger("vita.mcp")


def build_server(tool_layer: ToolLayer) -> Any:
    """Wrap a tool layer in an MCP server."""
    import mcp.types as types
    from mcp.server import Server

    server = Server("vita")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in tool_layer.list_tools()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        # Tool handlers are synchronous and some of them make network calls, so
        # run them off the event loop to keep the protocol responsive.
        result = await asyncio.to_thread(tool_layer.call, name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server


async def _serve() -> None:
    from mcp.server.stdio import stdio_server

    services = VitaServices()
    tool_layer = ToolLayer(services)
    server = build_server(tool_layer)

    print(
        f"[vita] MCP server ready - {APP_NAME} {APP_VERSION}, "
        f"mode={services.mode.value}, "
        f"tools={len(tool_layer.list_tools())}",
        file=sys.stderr,
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> int:
    parser = argparse.ArgumentParser(description="VITA MCP server")
    parser.parse_args()

    load_dotenv()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        print("[vita] shutting down", file=sys.stderr)
    except ImportError as exc:
        print(f"[vita] the mcp package is not installed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
