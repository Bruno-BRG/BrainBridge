// =======================================================
// BRAINBRIDGE - SETUP DE TESTE ESP32 (D22 / D23)
// Espelho do protocolo de Ortese/ortese.cpp, com saidas simples:
//   D23 = ESQUERDA
//   D22 = DIREITA
// Protocolo serial (115200 baud, 1 char — compativel com o backend):
//   'm' modo IA | 'l' pulso ESQUERDA | 'e' pulso DIREITA | 'o' parar
//   'z' ciclo automatico de teste | 'p' pausar/retomar | 'r' reset
//   'w'/'s' D23 ON/OFF manual | 'd'/'a' D22 ON/OFF manual
// =======================================================

#include <Arduino.h>

// --- Saidas de teste ---
const int pinoESQUERDA = 23; // D23
const int pinoDIREITA  = 22; // D22

// --- Tempos (ms), mesmos defaults da ortese ---
const unsigned long TEMPO_PULSO = 2000; // saida ativa por comando IA
const unsigned long TEMPO_PAUSA = 1000; // pausa do ciclo automatico

// --- Controle ---
bool sistemaCalibrado = false;
bool sistemaPausado   = false;
bool modoIA = false;
bool pulsoIAAtivo = false;
unsigned long tempoInicioIA = 0;

enum EstadoTeste {
  PULSO_ESQUERDA,
  PAUSA_POS_ESQUERDA,
  PULSO_DIREITA,
  PAUSA_POS_DIREITA
};

EstadoTeste estadoAtual = PULSO_ESQUERDA;
unsigned long tempoInicioEstado = 0;

void pararTudo() {
  digitalWrite(pinoESQUERDA, LOW);
  digitalWrite(pinoDIREITA, LOW);
}

void pulsarEsquerda() {
  digitalWrite(pinoESQUERDA, HIGH);
  digitalWrite(pinoDIREITA, LOW);
}

void pulsarDireita() {
  digitalWrite(pinoESQUERDA, LOW);
  digitalWrite(pinoDIREITA, HIGH);
}

void setup() {
  Serial.begin(115200);

  pinMode(pinoESQUERDA, OUTPUT);
  pinMode(pinoDIREITA, OUTPUT);

  pararTudo();
  exibirMenu();
}

void loop() {
  verificarComandosSerial();

  // Modo IA: um pulso por comando ('l'/'e'), desliga sozinho apos TEMPO_PULSO.
  if (modoIA && pulsoIAAtivo && !sistemaPausado) {
    if (millis() - tempoInicioIA >= TEMPO_PULSO) {
      pulsoIAAtivo = false;
      pararTudo();
      Serial.println(">> [IA] Pulso concluido");
    }
  }

  if (sistemaCalibrado && !sistemaPausado && !modoIA) {
    executarCicloTeste();
  }
}

// --- TRANSICAO DE ESTADOS (ciclo automatico) ---
void mudarEstado(EstadoTeste novoEstado) {
  estadoAtual = novoEstado;
  tempoInicioEstado = millis();

  switch (estadoAtual) {
    case PULSO_ESQUERDA:
      pulsarEsquerda();
      Serial.println(">> [TESTE] D23 ESQUERDA ativa");
      break;

    case PAUSA_POS_ESQUERDA:
      pararTudo();
      Serial.println(">> [PAUSA 1] Saidas desligadas");
      break;

    case PULSO_DIREITA:
      pulsarDireita();
      Serial.println(">> [TESTE] D22 DIREITA ativa");
      break;

    case PAUSA_POS_DIREITA:
      pararTudo();
      Serial.println(">> [PAUSA 2] Saidas desligadas");
      break;
  }
}

void executarCicloTeste() {
  unsigned long tempoDecorrido = millis() - tempoInicioEstado;

  switch (estadoAtual) {
    case PULSO_ESQUERDA:
      if (tempoDecorrido >= TEMPO_PULSO) mudarEstado(PAUSA_POS_ESQUERDA);
      break;
    case PAUSA_POS_ESQUERDA:
      if (tempoDecorrido >= TEMPO_PAUSA) mudarEstado(PULSO_DIREITA);
      break;
    case PULSO_DIREITA:
      if (tempoDecorrido >= TEMPO_PULSO) mudarEstado(PAUSA_POS_DIREITA);
      break;
    case PAUSA_POS_DIREITA:
      if (tempoDecorrido >= TEMPO_PAUSA) mudarEstado(PULSO_ESQUERDA);
      break;
  }
}

// --- LEITURA SERIAL (mesmo protocolo da ortese) ---
void verificarComandosSerial() {
  while (Serial.available() > 0) {
    char comando = Serial.read();

    if (comando == '\n' || comando == '\r' || comando == ' ') continue;
    comando = tolower(comando);

    if (!sistemaCalibrado) {
      switch (comando) {
        case 'w':
          Serial.println("Teste: D23 ESQUERDA ON");
          digitalWrite(pinoESQUERDA, HIGH); digitalWrite(pinoDIREITA, LOW);
          delay(200); pararTudo();
          break;

        case 's':
          Serial.println("Teste: D23 ESQUERDA OFF");
          pararTudo();
          break;

        case 'd':
          Serial.println("Teste: D22 DIREITA ON");
          digitalWrite(pinoDIREITA, HIGH); digitalWrite(pinoESQUERDA, LOW);
          delay(200); pararTudo();
          break;

        case 'a':
          Serial.println("Teste: D22 DIREITA OFF");
          pararTudo();
          break;

        case 'z':
          sistemaCalibrado = true;
          sistemaPausado   = false;
          modoIA = false;
          pulsoIAAtivo = false;
          Serial.println("\n>>> INICIANDO CICLO AUTOMATICO DE TESTE <<<");
          mudarEstado(PULSO_ESQUERDA);
          break;
      }
    } else {
      if (comando == 'm') {
        modoIA = true;
        pulsoIAAtivo = false;
        pararTudo();
        Serial.println("\n>>> MODO IA: aguardando comandos 'l' (esquerda/D23) / 'e' (direita/D22) <<<");
      } else if (comando == 'l') {
        if (modoIA && !sistemaPausado) {
          pulsoIAAtivo = false;
          pulsarEsquerda();
          pulsoIAAtivo = true;
          tempoInicioIA = millis();
          Serial.println(">> [IA] ESQUERDA (D23) sob demanda");
        }
      } else if (comando == 'e') {
        if (modoIA && !sistemaPausado) {
          pulsoIAAtivo = false;
          pulsarDireita();
          pulsoIAAtivo = true;
          tempoInicioIA = millis();
          Serial.println(">> [IA] DIREITA (D22) sob demanda");
        }
      } else if (comando == 'o') {
        pulsoIAAtivo = false;
        pararTudo();
        Serial.println(">> [IA] Parado");
      } else if (comando == 'z') {
        modoIA = false;
        pulsoIAAtivo = false;
        sistemaPausado = false;
        Serial.println("\n>>> INICIANDO CICLO AUTOMATICO DE TESTE <<<");
        mudarEstado(PULSO_ESQUERDA);
      }
    }

    if (comando == 'p') {
      sistemaPausado = !sistemaPausado;
      if (sistemaPausado) {
        pulsoIAAtivo = false;
        pararTudo();
        Serial.println("|| SISTEMA PAUSADO");
      } else {
        tempoInicioEstado = millis();
        Serial.println(">> RETOMANDO");
        if (modoIA) {
          Serial.println(">> MODO IA ativo");
        } else {
          mudarEstado(estadoAtual);
        }
      }
    }

    if (comando == 'r') {
      sistemaCalibrado = false;
      sistemaPausado   = false;
      modoIA = false;
      pulsoIAAtivo = false;
      pararTudo();
      Serial.println("\n!!! RESET: Modo de teste manual");
      exibirMenu();
    }
  }
}

void exibirMenu() {
  Serial.println("\n=============================================");
  Serial.println("    BRAINBRIDGE - SETUP DE TESTE D22/D23");
  Serial.println("=============================================");
  Serial.println("D23 = ESQUERDA | D22 = DIREITA");
  Serial.println("[ W / S ] Testar D23 (esquerda ON/OFF)");
  Serial.println("[ D / A ] Testar D22 (direita ON/OFF)");
  Serial.println("[   Z   ] Ciclo automatico de teste");
  Serial.println("[   M   ] Modo IA (comandos sob demanda)");
  Serial.println("[ L / E ] IA: esquerda / direita (modo IA)");
  Serial.println("[   O   ] IA: desligar saidas");
  Serial.println("[   P   ] Pausar / Retomar");
  Serial.println("[   R   ] Reset");
  Serial.println("=============================================\n");
}
