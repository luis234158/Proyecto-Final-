/*
  Clasificador de monedas - lado ESP32
  ---------------------------------------------------------------------------
  Este sketch es la contraparte de detector.py (el que corre en el PC con YOLO).
  El PC detecta la moneda y manda ordenes por el cable USB (serial, 115200).
  El ESP32 mueve la banda (motorreductor con driver) y posiciona el servo.

  Comandos que entiende (una linea terminada en \n), igual que detector.py:
    B,<vel>        banda continua a vel % (0 = parar)             -> OK,B
    T,<vel>,<ms>   banda a vel % durante ms y se detiene sola      -> OK,T ... FIN,T
    A,<angulo>     servo a un angulo (0-180)                       -> OK,A
    C,<clase>      servo a la posicion de esa moneda               -> OK,C
                   (clase = 50, 100, 200, 500, 1000 o rechazo)
    E              prueba de vida                                  -> E,OK
  Al arrancar manda  LISTO  (detector.py espera esa palabra).
  El OK de A y C llega cuando el servo ya tuvo tiempo de llegar.

  Conexiones (ESP32 clasico, DevKit). Cambialas aqui abajo si usas otras:
    GPIO 25  -> ENA / PWMA del driver (velocidad)
    GPIO 26  -> IN1 del driver        (sentido)
    GPIO 27  -> IN2 del driver        (sentido)
    GPIO 18  -> cable de senal del servo (naranja/amarillo)
    GND del ESP32, del driver y del servo unidos.
    El servo y el motor van con su propia fuente, NO con el 3V3 del ESP32.
    Si usas un ESP32-S3 cambia los pines por 4, 5, 6 y 7.
    Recomendado: resistencia de 10 k a GND en IN1, IN2 y ENA, para que el
    motor no se mueva solo mientras el ESP32 se reinicia.

  Como subirlo:
    - Arduino IDE: placa "ESP32 Dev Module", abre este archivo y Subir.
      No necesita librerias extra.
    - VS Code + PlatformIO: pega el contenido en src/main.cpp (ver
      platformio.ini en la respuesta) y pulsa Upload.
  Cierra el Monitor Serie antes de correr detector.py: el puerto COM solo lo
  puede usar un programa a la vez.
*/

#include <Arduino.h>

// ---------------- Pines ----------------
const int PIN_MOTOR_PWM = 25;
const int PIN_MOTOR_IN1 = 26;
const int PIN_MOTOR_IN2 = 27;
const int PIN_SERVO     = 18;

// true si la banda se mueve al reves de lo que quieres
const bool INVERTIR_BANDA = false;

// ---------------- Servo: posiciones (CALIBRAR) ----------------
const int ANG_INICIO = 90;           // posicion al encender
struct PosMoneda { const char* clase; int angulo; };
const PosMoneda POSICIONES[] = {
  {"50",      12},
  {"100",     46},
  {"200",     74},
  {"500",    112},
  {"1000",   148},
  {"rechazo", 176},                  // tambien se usa si llega una clase desconocida
};
const int N_POSICIONES = sizeof(POSICIONES) / sizeof(POSICIONES[0]);
const int ANG_RECHAZO = 176;

// Pulsos del servo en microsegundos (SG90/MG90: 500-2400; si "pega" en los
// extremos y vibra, acorta el rango, por ejemplo 600-2300)
const int SERVO_US_MIN = 500;
const int SERVO_US_MAX = 2400;

// ---------------- PWM (canales LEDC) ----------------
// Canal 0 y canal 2 usan temporizadores distintos (necesario porque una
// frecuencia es de 20 kHz y la otra de 50 Hz).
const int CH_MOTOR = 0, CH_SERVO = 2;
const int MOTOR_FREQ = 20000, MOTOR_BITS = 10;   // 0..1023, silencioso
const int SERVO_FREQ = 50,    SERVO_BITS = 14;

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
void pwmInit(int pin, int ch, int freq, int bits) { ledcAttachChannel(pin, freq, bits, ch); }
void pwmWrite(int pin, int, uint32_t duty)        { ledcWrite(pin, duty); }
#else   // core 2.x (el que usa PlatformIO por defecto)
void pwmInit(int pin, int ch, int freq, int bits) { ledcSetup(ch, freq, bits); ledcAttachPin(pin, ch); }
void pwmWrite(int, int ch, uint32_t duty)         { ledcWrite(ch, duty); }
#endif

// ---------------- Estado ----------------
int  velBanda = 0;
int  angServo = ANG_INICIO;
bool banda_en_tiempo = false;        // true durante un comando T
unsigned long tFinBanda = 0;

bool respPendiente = false;          // OK,A / OK,C que se manda cuando el servo llega
char respTexto[8] = "";
unsigned long tResp = 0;

char linea[48];
int  nLinea = 0;
bool descartando = false;

// ---------------- Actuadores ----------------
void motor(int vel) {
  vel = constrain(vel, 0, 100);
  bool avance = !INVERTIR_BANDA;
  digitalWrite(PIN_MOTOR_IN1, (vel > 0 && avance)  ? HIGH : LOW);
  digitalWrite(PIN_MOTOR_IN2, (vel > 0 && !avance) ? HIGH : LOW);
  pwmWrite(PIN_MOTOR_PWM, CH_MOTOR, (uint32_t)vel * ((1UL << MOTOR_BITS) - 1) / 100);
  velBanda = vel;
}

// Mueve el servo y devuelve los ms que tarda en llegar (estimado)
unsigned long servo(int ang) {
  ang = constrain(ang, 0, 180);
  uint32_t us = SERVO_US_MIN + (uint32_t)(SERVO_US_MAX - SERVO_US_MIN) * ang / 180;
  pwmWrite(PIN_SERVO, CH_SERVO, us * ((1UL << SERVO_BITS) - 1) / 20000UL);
  unsigned long ms = 150 + (unsigned long)abs(ang - angServo) * 4;
  angServo = ang;
  return ms;
}

int anguloDeClase(const char* clase) {
  for (int i = 0; i < N_POSICIONES; i++)
    if (strcmp(clase, POSICIONES[i].clase) == 0) return POSICIONES[i].angulo;
  return ANG_RECHAZO;
}

void responderCuando(const char* texto, unsigned long ms) {
  if (respPendiente) Serial.println(respTexto);      // no pisar una respuesta anterior
  strncpy(respTexto, texto, sizeof(respTexto) - 1);
  respTexto[sizeof(respTexto) - 1] = 0;
  tResp = millis() + ms;
  respPendiente = true;
}

// ---------------- Comandos ----------------
void error(const char* l) {
  Serial.print("ERR,");
  Serial.println(l);
}

void procesar(char* l) {
  while (*l == ' ') l++;
  if (*l == 0) return;
  int v;
  long ms;
  switch (l[0]) {
    case 'B':
      if (sscanf(l, "B,%d", &v) == 1) {
        banda_en_tiempo = false;
        motor(v);
        Serial.println("OK,B");
      } else error(l);
      break;
    case 'T':
      if (sscanf(l, "T,%d,%ld", &v, &ms) == 2 && ms > 0) {
        motor(v);
        banda_en_tiempo = true;
        tFinBanda = millis() + (unsigned long)ms;
        Serial.println("OK,T");
      } else error(l);
      break;
    case 'A':
      if (sscanf(l, "A,%d", &v) == 1) responderCuando("OK,A", servo(v));
      else error(l);
      break;
    case 'C':
      if (l[1] == ',' && l[2] != 0) {
        char* clase = l + 2;
        responderCuando("OK,C", servo(anguloDeClase(clase)));
      } else error(l);
      break;
    case 'E':
      Serial.println("E,OK");
      break;
    default:
      error(l);
  }
}

void leerSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (!descartando && nLinea > 0) { linea[nLinea] = 0; procesar(linea); }
      nLinea = 0; descartando = false;
    } else if (nLinea < (int)sizeof(linea) - 1) {
      linea[nLinea++] = c;
    } else {
      descartando = true;                // linea demasiado larga: se ignora completa
    }
  }
}

// ---------------- Setup / Loop ----------------
void setup() {
  pinMode(PIN_MOTOR_IN1, OUTPUT);
  pinMode(PIN_MOTOR_IN2, OUTPUT);
  digitalWrite(PIN_MOTOR_IN1, LOW);
  digitalWrite(PIN_MOTOR_IN2, LOW);
  pwmInit(PIN_MOTOR_PWM, CH_MOTOR, MOTOR_FREQ, MOTOR_BITS);
  pwmInit(PIN_SERVO, CH_SERVO, SERVO_FREQ, SERVO_BITS);
  motor(0);
  servo(ANG_INICIO);
  Serial.begin(115200);
  Serial.println("LISTO");
}

void loop() {
  leerSerial();
  unsigned long ahora = millis();

  if (banda_en_tiempo && (long)(ahora - tFinBanda) >= 0) {   // termino la entrega
    motor(0);
    banda_en_tiempo = false;
    Serial.println("FIN,T");
  }
  if (respPendiente && (long)(ahora - tResp) >= 0) {         // el servo ya llego
    Serial.println(respTexto);
    respPendiente = false;
  }
}
