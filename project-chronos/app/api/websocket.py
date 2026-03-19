"""
WebSocket Connection Manager for Project Chronos.

Manages multiple connected clients and broadcasts patient state
updates and alert events in real-time.

Message format:
  { "event": "patient_update", "data": { ...PatientState... } }
  { "event": "new_alert", "data": { ...AlertObject... } }
  { "event": "system_status", "data": { ...status... } }
"""

import asyncio
import json
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts messages.

    Thread-safe for use with FastAPI's async architecture.
    Handles client disconnections gracefully.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Remove a disconnected client."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to ALL connected clients.
        Silently removes clients that have disconnected.
        """
        if not self.active_connections:
            return

        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)

        if dead:
            async with self._lock:
                for conn in dead:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)

    async def broadcast_patient_update(self, patient_state_dict: Dict):
        """Broadcast a patient state update event."""
        await self.broadcast({
            "event": "patient_update",
            "data": patient_state_dict,
        })

    async def broadcast_alert(self, alert_dict: Dict):
        """Broadcast a new alert event."""
        await self.broadcast({
            "event": "new_alert",
            "data": alert_dict,
        })

    async def broadcast_status(self, status: Dict):
        """Broadcast system status."""
        await self.broadcast({
            "event": "system_status",
            "data": status,
        })

    @property
    def client_count(self) -> int:
        return len(self.active_connections)
