#include <ESP8266WiFi.h>
#include <Wire.h>
#include <MPU6050.h>

const char *ssid = "andy";
const char *password = "123123123";

WiFiServer server(80);

MPU6050 mpu;
float roll, pitch, yaw;

// Keep last N readings in memory
#define MAX_READINGS 200
String readings[MAX_READINGS];
int readingIndex = 0;

void setup()
{
    Serial.begin(115200);
    Wire.begin(4, 5); // SDA, SCL

    mpu.initialize();

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    Serial.print("Connecting to Wi-Fi");
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    server.begin();
}

void loop()
{
    // Read MPU6050 data
    float rollSum = 0, pitchSum = 0, yawSum = 0;

    for (int i = 0; i < 100; i++)
    {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        rollSum += ax / 16384.0;
        pitchSum += ay / 16384.0;
        yawSum += gz / 131.0;

        delay(5); // small delay to avoid hammering the bus
    }

    // average
    roll = rollSum / 100;
    pitch = pitchSum / 100;
    yaw = yawSum / 100;

    // Add new reading to the log
    String response = "Roll: " + String(roll) +
                      " | Pitch: " + String(pitch) +
                      " | Yaw: " + String(yaw);
    readings[readingIndex] = response;
    readingIndex = (readingIndex + 1) % MAX_READINGS; // circular buffer

    // Serve client
    WiFiClient client = server.available();
    if (client)
    {
        client.println("HTTP/1.1 200 OK");
        client.println("Content-Type: text/plain");
        client.println("Connection: close");
        client.println();

        // Output all readings sequentially
        for (int i = 0; i < MAX_READINGS; i++)
        {
            int idx = (readingIndex + i) % MAX_READINGS; // wrap around
            if (readings[idx].length() > 0)
            {
                client.println(readings[idx]);
            }
        }

        delay(1);
    }

    delay(100); // adjust logging rate
}
