import asyncio, json, logging
log = logging.getLogger("kinetimesh.mqtt")

class MQTTService:
    """MQTT service stub — replace with aiomqtt in production"""
    def __init__(self): self.subscribers = {}
    async def publish(self, topic: str, payload: dict):
        log.debug(f"MQTT publish {topic}: {json.dumps(payload)[:80]}")
    async def subscribe(self, topic: str, callback):
        self.subscribers[topic] = callback

mqtt_service = MQTTService()
