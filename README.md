# AI-Driver-Detection-System

The primary objective of this project was to deliver a practical and physical solution to a common, real-life problem while integrating basic AI applications with a hardware base.

It primarily uses MediaPipe Solutions, from Google AI Edge, libraries as a quick way to implement AI and this helps in tracking various things like MAR (Mouth Aspect Ratio), EAR (Eye Aspect Ratio) and Head pose angles specifically the head yaw and pitch angles and when they exceed or fall below the required amounts, it sends a signal to trigger the buzzer and displays the actual condition if the driver is drowsy or distracted. 

The Arduino uses a ultrasonic sensor to see if the distance between the driver and the sensor goes too low and wheter the driver has fallen asleep or fainted, while also serving as a trigger for the buzzer to wake up the driver.
