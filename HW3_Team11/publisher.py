import paho.mqtt.client as mqtt
import uuid
import platform
import psutil
import time
import json

# Set the MQTT topic
MQTT_TOPIC = "## ADD HERE THE TOPIC YOU WANT TO PUBLISH FROM ##"

# Get the MAC address of the PC
mac_address =  hex(uuid.getnode())
print(mac_address) 

# Create the MQTT client
client = mqtt.Client()

# Connect to the MQTT broker
client.connect('mqtt.eclipseprojects.io', 1883) #The MQTT broker can be changed

while True:
    # Get the current time in milliseconds
    timestamp = int(time.time() * 1000) #from the source we want the timestamp in milliseconds

    # Get the battery level and power plugged status
    battery = psutil.sensors_battery()
    battery_level = battery.percent
    power_plugged = 1 if battery.power_plugged else 0

    # Create the message payload as a JSON object
    message = {
        "mac_address": mac_address,
        "timestamp": timestamp,
        "battery_level": battery_level,
        "power_plugged": power_plugged
    }

    # Convert the message payload to a JSON string
    message_json = json.dumps(message)
    print(message_json)
    # Publish the message to the MQTT topic
    client.publish(MQTT_TOPIC, message_json)

    # Sleep for 1 second
    time.sleep(1)


