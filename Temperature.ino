#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <SparkFun_APDS9960.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels
#define OLED_RESET    4 // Reset pin # (or -1 if sharing Arduino reset pin)

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET); //Declaring the display name (display)
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

SparkFun_APDS9960 apds = SparkFun_APDS9960();
uint8_t proximity_data = 0;

const int buzzer = 9; //buzzer to arduino pin 

void setup() {  
  mlx.begin(); 
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C); //Start the OLED display
  display.clearDisplay();
  display.display();
  apds.init();
  apds.enableProximitySensor(false);
  pinMode(buzzer, OUTPUT); // Set buzzer - pin 9 as an output

}

void loop() {
  String temperature = "";
  display.clearDisplay();
  
  display.setTextSize(1);                    
  display.setTextColor(WHITE);             
  display.setCursor(0,4);                
  display.println("Ambient : "); 
  
  display.setTextSize(2);
  display.setCursor(50,0);
  display.println(mlx.readAmbientTempF(),1);
 
  display.setCursor(110,0);
  display.println("F");
  
  display.setTextSize(1);                    
  display.setTextColor(WHITE);             
  display.setCursor(0,20);                
  display.println("Target : "); 
  
  display.setTextSize(2);
  display.setCursor(50,17);
  display.println(8 + mlx.readObjectTempF(),1);
  
  display.setCursor(110,17);
  display.println("F");
  
  display.display();
  display.clearDisplay();
  delay(1000);

  if((8 + mlx.readObjectTempF()) > 100){
    
    display.clearDisplay();
    display.invertDisplay(false);
    display.setTextSize(2);                    
    display.setTextColor(WHITE);             
    display.setCursor(0,4);                
    display.print("CRITICAL");
    tone(buzzer, 1000); // Send 1KHz sound signal...
    delay(1000);        // ...for 1 sec
    noTone(buzzer);     // Stop sound...
    delay(1000);        // ...for 1sec    
    }

  else if((8 + mlx.readObjectTempF()) < 100){
    
    display.clearDisplay();
    display.invertDisplay(true);
    display.setTextSize(2);                    
    display.setTextColor(WHITE);             
    display.setCursor(0,4);
    display.print("NORMAL");  
    }

//  if (proximity_data == 5 && mlx.readObjectTempF() < 100) {
//
//
//    temperature = String(mlx.readObjectTempF(), 1);
//    Serial.print(" Body Temperature:");
//    Serial.println(mlx.readObjectTempF());
//    display.clearDisplay();
//    display.invertDisplay(false);
//    display.setTextSize(2);
//    display.setTextColor(WHITE);
//    display.setCursor(8, 0);
//    display.clearDisplay();
//    display.println("Body Temp:");
//    display.setCursor(25, 18);
//    display.print(mlx.readObjectTempF());
//
//    display.setCursor(85, 8);
//    display.print(".");
//
//
//    display.setTextColor(WHITE);
//    display.setCursor(85, 18);
//    display.print(" F");
//    display.display();
//
//    delay(1000);
//
//  }
//  if (proximity_data == 5) {
//    if (mlx.readObjectTempF() > 102) {
//      noTone(5);
//      // play a note on pin 8 for 500 ms:
//      tone(8, 523, 500);
//
//      display.clearDisplay();
//      display.invertDisplay(false);
//      display.setTextSize(2);
//      display.setTextColor(WHITE);
//      display.setCursor(20, 0);
//      display.clearDisplay();
//      display.println("CRITICAL");
//
//      display.invertDisplay(true);
//      display.setTextSize(2);
//      display.setTextColor(WHITE);
//      display.setCursor(20, 0);
//      display.clearDisplay();
//      display.println("CRITICAL");
//
//
//      display.invertDisplay(true);
//      display.setTextColor(WHITE);
//      display.setCursor(20, 0);
//      display.clearDisplay();
//      display.println("CRITICAL");
//
//
//      display.invertDisplay(false);
//      display.setTextSize(2);
//      display.setTextColor(WHITE);
//      display.setCursor(20, 0);
//      display.clearDisplay();
//      display.println("CRITICAL");
//
//      display.invertDisplay(true);
//      display.setTextSize(2);
//      display.setTextColor(WHITE);
//      display.setCursor(20, 0);
//      display.clearDisplay();
//      display.println("CRITICAL");
//
//
//      display.invertDisplay(true);
//      display.setTextColor(WHITE);
//      display.setCursor(20, 0);
//      display.clearDisplay();
//      display.println("CRITICAL");
//
//
//      display.invertDisplay(false);
//      display.setTextSize(2);
//      display.setTextColor(WHITE);
//      display.setCursor(20, 0);
//      display.clearDisplay();
//      display.println("CRITICAL");
//
//
//
//
//      display.setCursor(23, 18);
//      display.print(mlx.readObjectTempF());
//      display.setCursor(93, 8);
//      display.print(".");
//      display.setTextColor(WHITE);
//      display.setCursor(93, 18);
//      display.print(" F");
//      display.display();
//
//
//      delay(1000);
//    }
//
//  }
//
//  if (proximity_data == 5) {
//    if (mlx.readObjectTempF() > 100) {
//
//      display.setCursor(93, 8);
//      display.print(".");
//
//
//      display.setTextColor(WHITE);
//      display.setCursor(93, 18);
//      display.print(" F");
//      display.display();
//
//
//
//    }
//  }
//
//
//  else if (proximity_data <= 5) {
//    delay(1000);
//
//    display.clearDisplay();
//    display.invertDisplay(true);
//    display.setTextSize(2.8);
//    display.setTextColor(WHITE);
//    display.setCursor(45, 1);
//    display.clearDisplay();
//    display.println("NOT");
//    display.setCursor(17, 17);
//    display.print("DETECTED");
//    display.display();
//    digitalWrite(5, LOW);
//
//  }

}
