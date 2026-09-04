"""
The MCP session the planner talks through.

MCP is in the request path, not beside it. When a patient sends a message, the
planner reaches every capability it has by calling a tool over an MCP session -
`list_tools`, `call_tool`, JSON-RPC, the lot. Nothing about that is simulated.

What is unusual is the transport. The obvious way to run an MCP server is as a
subprocess over stdio, and that is exactly what `src/mcp_server.py` does for
external clients. Doing the same for our own planner would mean a second process
holding a second copy of the knowledge base, a second SQLite connection and a
second in-memory case cache - two systems disagreeing about the same patient.

So the in-process planner uses MCP's in-memory transport instead: a genuine
client and server connected by memory streams rather than pipes. Same protocol,
same server object, same tool implementations, one process and one set of state.
The stdio server remains for anyone driving VITA from outside.

The session is opened once at startup and lives for the life of the application.
Opening one per request would pay the initialisation handshake on every patient
message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class McpBridge:
    """A long-lived MCP client session against VITA's own tool server.

    Presents a synchronous interface because the rest of the application is
    synchronous. The event loop runs on a dedicated thread and every call is
    marshalled onto it, so FastAPI's worker threads can use MCP without each
    needing a loop of its own.
    """

    def __init__(self, server_factory: Any) -> None:
        self._server_factory = server_factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._exit_stack: Any = None
        self._tools: list[dict[str, Any]] = []
        self._unavailable_reason = ""
        self._start()

    # -- lifecycle -------------------------------------------------------

    def _start(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop, name="vita-mcp", daemon=True
            )
            self._thread.start()
            self._call_async(self._connect(), timeout=30)
            logger.info("MCP session ready: %d tools", len(self._tools))
        except Exception as exc:  # noqa: BLE001 - startup must not fail here
            self._unavailable_reason = f"could not open MCP session: {exc}"
            logger.error(self._unavailable_reason)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        from contextlib import AsyncExitStack

        from mcp.shared.memory import create_connected_server_and_client_session

        server = self._server_factory()
        self._exit_stack = AsyncExitStack()
        self._session = await self._exit_stack.enter_async_context(
            create_connected_server_and_client_session(server)
        )
        listing = await self._session.list_tools()
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema or {"type": "object", "properties": {}},
            }
            for tool in listing.tools
        ]

    def close(self) -> None:
        if self._loop is None:
            return
        try:
            if self._exit_stack is not None:
                self._call_async(self._exit_stack.aclose(), timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -- state -----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._session is not None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "transport": "in-memory (same process)",
            "tools": len(self._tools),
            "reason": self._unavailable_reason,
        }

    def tools(self) -> list[dict[str, Any]]:
        """The tool surface, as the server advertises it over the protocol."""
        return list(self._tools)

    # -- calling ---------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
        """Invoke a tool and return its result.

        Errors come back as data, never as exceptions. A planner told "no case
        with that id" corrects itself on the next step; one handed a traceback
        stops working.
        """
        if not self.available:
            return {"error": self._unavailable_reason or "MCP session unavailable"}

        try:
            return self._call_async(self._call_tool(name, arguments or {}), timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP call %s failed", name)
            return {"error": f"{type(exc).__name__}: {exc}"}

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._session.call_tool(name, arguments)
        blocks = [b for b in (result.content or []) if getattr(b, "text", None)]
        if not blocks:
            return {"error": f"tool {name} returned no content"}
        try:
            return json.loads(blocks[0].text)
        except json.JSONDecodeError:
            return {"result": blocks[0].text}

    def _call_async(self, coro: Any, *, timeout: float) -> Any:
        if self._loop is None:
            raise RuntimeError("MCP event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)
