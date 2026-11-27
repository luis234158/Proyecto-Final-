#include <Arduino.h>

const int PIN_MOTOR_PWM = 25;
const int PIN_MOTOR_IN1 = 26;
const int PIN_MOTOR_IN2 = 27;
const int PIN_SERVO     = 18;

const bool INVERTIR_BANDA = false;

const int ANG_INICIO = 90;
struct PosMoneda { const char* clase; int angulo; };
const PosMoneda POSICIONES[] = {
  {"50",      12},
  {"100",     46},
  {"200",     74},
  {"500",    112},
  {"1000",   148},
  {"rechazo", 176},
};
const int N_POSICIONES = sizeof(POSICIONES) / sizeof(POSICIONES[0]);
const int ANG_RECHAZO = 176;

const int SERVO_US_MIN = 500;
const int SERVO_US_MAX = 2400;

const int CH_MOTOR = 0, CH_SERVO = 2;
const int MOTOR_FREQ = 20000, MOTOR_BITS = 10;
const int SERVO_FREQ = 50,    SERVO_BITS = 14;

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
void pwmInit(int pin, int ch, int freq, int bits) { ledcAttachChannel(pin, freq, bits, ch); }
void pwmWrite(int pin, int, uint32_t duty)        { ledcWrite(pin, duty); }
#else
void pwmInit(int pin, int ch, int freq, int bits) { ledcSetup(ch, freq, bits); ledcAttachPin(pin, ch); }
void pwmWrite(int, int ch, uint32_t duty)         { ledcWrite(ch, duty); }
#endif

int  velBanda = 0;
int  angServo = ANG_INICIO;
bool banda_en_tiempo = false;
unsigned long tFinBanda = 0;

bool respPendiente = false;
char respTexto[8] = "";
unsigned long tResp = 0;

char linea[48];
int  nLinea = 0;
bool descartando = false;

void motor(int vel) {
  vel = constrain(vel, 0, 100);
  bool avance = !INVERTIR_BANDA;
  digitalWrite(PIN_MOTOR_IN1, (vel > 0 && avance)  ? HIGH : LOW);
  digitalWrite(PIN_MOTOR_IN2, (vel > 0 && !avance) ? HIGH : LOW);
  pwmWrite(PIN_MOTOR_PWM, CH_MOTOR, (uint32_t)vel * ((1UL << MOTOR_BITS) - 1) / 100);
  velBanda = vel;
}

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
  if (respPendiente) Serial.println(respTexto);
  strncpy(respTexto, texto, sizeof(respTexto) - 1);
  respTexto[sizeof(respTexto) - 1] = 0;
  tResp = millis() + ms;
  respPendiente = true;
}

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
      descartando = true;
    }
  }
}

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

  if (banda_en_tiempo && (long)(ahora - tFinBanda) >= 0) {
    motor(0);
    banda_en_tiempo = false;
    Serial.println("FIN,T");
  }
  if (respPendiente && (long)(ahora - tResp) >= 0) {
    Serial.println(respTexto);
    respPendiente = false;
  }
}
