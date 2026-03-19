#!/usr/bin/env python3
"""
Simple WebSocket client to test Chronos real-time updates.
"""

import asyncio
import websockets
import json

async def test_chronos_websocket():
    uri = 'ws://localhost:8000/ws'
    
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Chronos WebSocket!")
            
            # Listen for messages
            message_count = 0
            async for message in websocket:
                data = json.loads(message)
                print(f"📨 Message {message_count + 1}:")
                print(f"   Event: {data.get('event', 'unknown')}")
                print(f"   Data: {json.dumps(data.get('data', {}), indent=2)}")
                print()
                
                message_count += 1
                
                # Stop after 10 messages or 30 seconds
                if message_count >= 10:
                    print("✅ Received 10 messages - test complete!")
                    break
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_chronos_websocket())
