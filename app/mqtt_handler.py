import paho.mqtt.client as mqtt
import datetime

class MqttHandler:
    '''
    When you send mqtt CONNECT packet, you should receive CONNACK response. This response contains the following codes
    Connection Return Codes

    0: Connection successful
    1: Connection refused – incorrect protocol version
    2: Connection refused – invalid client identifier
    3: Connection refused – server unavailable
    4: Connection refused – bad username or password
    5: Connection refused – not authorised
    '''

    def __init__(self, on_message_callback):
        self.on_message_callback = on_message_callback

        self.client = mqtt.Client(client_id="heimdall-backend")
        self.client.on_connect =   self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message =   self.on_message
        self.client.on_log =       self.on_log
        self.client.on_publish =   self.on_publish
        self.client.on_subscribe = self.on_subscribe

    def connect(self):
        # Create mqtt-broker in docker
        self.client.connect("mqtt-broker", 1883, 60)
        self.client.loop_start()
        #self.client.loop_forever()

    def disconnect(self):
        self.client.disconnect()
        self.client.loop_stop(force=False) # TODO stop mqtt on close

    # The callback for when the client receives a CONNACK response from the server.
    @staticmethod
    def on_connect(client, userdata, flags, rc):
        print("=========")
        print("Connected:", mqtt.connack_string(rc))

        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        #client.subscribe("$SYS/#")
        client.subscribe("World")

        # TODO subscribe to correct topics

    @staticmethod
    def on_disconnect(client, userdata, rc):
        print("=========")
        print("Disconnected:", mqtt.connack_string(rc))

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print("=========")
        print(msg.topic+" "+str(msg.payload))

        self.on_message_callback(client, userdata, msg)


    @staticmethod
    def on_subscribe(mosq, obj, mid, granted_qos):
        print("Subscribed to Topic with QoS: " + str(granted_qos))


    @staticmethod
    def on_publish(client, obj, mid):
        print("=========")
        print("Publish!!")
        print(obj)
        print(mid)

    @staticmethod
    def on_log(client, userdata, level, buf):
        print(level, buf)



