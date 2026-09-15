// =======================================================
// SISTEMA DE ÓRTESE ROBÓTICA - CIMATEC
// ROTINA AUTOMÁTICA DE FISIOTERAPIA COM CARRETEL DUPLO
// =======================================================

// --- Pinos Motor 1: Extensão (E) ---
const int pinoIN1_E = 27;
const int pinoIN2_E = 14;

// --- Pinos Motor 2: Flexão (F) ---
const int pinoIN3_F = 13;
const int pinoIN4_F = 12;

// --- Configuração dos Tempos de Fisioterapia (em milissegundos) ---
const unsigned long TEMPO_MOVIMENTO = 2000; // 2 segundos girando
const unsigned long TEMPO_PAUSA     = 1000; // 1 segundo parado

// --- Variáveis de Controle ---
bool sistemaCalibrado = false;
bool sistemaPausado   = false;
// Modo IA (BrainBridge Livre/Jogo): movimentos sob demanda via 'l'/'e'.
// NAO altera nenhum output de motor: reusa moverFlexao/moverExtensao/
// pararTudo e os mesmos TEMPO_MOVIMENTO/TEMPO_PAUSA.
bool modoIA = false;
bool movimentoIAAtivo = false;
unsigned long tempoInicioIA = 0;

enum EstadoFisioterapia {
  MOVENDO_FLEXAO,
  PAUSA_POS_FLEXAO,
  MOVENDO_EXTENSAO,
  PAUSA_POS_EXTENSAO
};

EstadoFisioterapia estadoAtual = MOVENDO_FLEXAO;
unsigned long tempoInicioEstado = 0;

void setup() {
  Serial.begin(115200);

  pinMode(pinoIN1_E, OUTPUT);
  pinMode(pinoIN2_E, OUTPUT);
  pinMode(pinoIN3_F, OUTPUT);
  pinMode(pinoIN4_F, OUTPUT);

  pararTudo();
  exibirMenu();
}

void loop() {
  verificarComandosSerial();

  // Modo IA: um movimento por comando ('l'/'e'), com parada automatica
  // apos TEMPO_MOVIMENTO. Sem ciclo automatico aqui.
  if (modoIA && movimentoIAAtivo && !sistemaPausado) {
    if (millis() - tempoInicioIA >= TEMPO_MOVIMENTO) {
      movimentoIAAtivo = false;
      pararTudo();
      Serial.println(">> [IA] Movimento concluido");
    }
  }

  if (sistemaCalibrado && !sistemaPausado && !modoIA) {
    executarFisioterapia();
  }
}

// --- CONTROLE DOS MOTORES ---
void moverFlexao() {
  // Motor 2 puxa (Direita) / Motor 1 desenrola (Direita)
  digitalWrite(pinoIN1_E, HIGH);
  digitalWrite(pinoIN2_E, LOW);
  digitalWrite(pinoIN3_F, HIGH);
  digitalWrite(pinoIN4_F, LOW);
}

void moverExtensao() {
  // Motor 1 puxa (Esquerda) / Motor 2 desenrola (Esquerda)
  digitalWrite(pinoIN1_E, HIGH);
  digitalWrite(pinoIN2_E, LOW);
  digitalWrite(pinoIN3_F, HIGH);
  digitalWrite(pinoIN4_F, LOW);
}

void pararTudo() {
  digitalWrite(pinoIN1_E, LOW);
  digitalWrite(pinoIN2_E, LOW);
  digitalWrite(pinoIN3_F, LOW);
  digitalWrite(pinoIN4_F, LOW);
}

// --- TRANSIÇÃO DE ESTADOS ---
void mudarEstado(EstadoFisioterapia novoEstado) {
  estadoAtual = novoEstado;
  tempoInicioEstado = millis();

  switch (estadoAtual) {
    case MOVENDO_FLEXAO:
      moverFlexao();
      Serial.println(">> [FLEXÃO] Motores em movimento (Direita)");
      break;

    case PAUSA_POS_FLEXAO:
      pararTudo();
      Serial.println(">> [PAUSA 1] Motores parados");
      break;

    case MOVENDO_EXTENSAO:
      moverExtensao();
      Serial.println(">> [EXTENSÃO] Motores em movimento (Esquerda)");
      break;

    case PAUSA_POS_EXTENSAO:
      pararTudo();
      Serial.println(">> [PAUSA 2] Motores parados");
      break;
  }
}

// --- ROTINA AUTOMÁTICA ---
void executarFisioterapia() {
  unsigned long tempoDecorrido = millis() - tempoInicioEstado;

  switch (estadoAtual) {
    case MOVENDO_FLEXAO:
      if (tempoDecorrido >= TEMPO_MOVIMENTO) {
        mudarEstado(PAUSA_POS_FLEXAO);
      }
      break;

    case PAUSA_POS_FLEXAO:
      if (tempoDecorrido >= TEMPO_PAUSA) {
        mudarEstado(MOVENDO_EXTENSAO);
      }
      break;

    case MOVENDO_EXTENSAO:
      if (tempoDecorrido >= TEMPO_MOVIMENTO) {
        mudarEstado(PAUSA_POS_EXTENSAO);
      }
      break;

    case PAUSA_POS_EXTENSAO:
      if (tempoDecorrido >= TEMPO_PAUSA) {
        mudarEstado(MOVENDO_FLEXAO);
      }
      break;
  }
}

// --- LEITURA SERIAL ---
void verificarComandosSerial() {
  while (Serial.available() > 0) {
    char comando = Serial.read();

    if (comando == '\n' || comando == '\r' || comando == ' ') continue;
    comando = tolower(comando);

    if (!sistemaCalibrado) {
      switch (comando) {
        case 'w':
          Serial.println("Calibração: Motor 1 Direita");
          digitalWrite(pinoIN1_E, HIGH); digitalWrite(pinoIN2_E, LOW);
          delay(200); pararTudo();
          break;

        case 's':
          Serial.println("Calibração: Motor 1 Esquerda");
          digitalWrite(pinoIN1_E, LOW); digitalWrite(pinoIN2_E, HIGH);
          delay(200); pararTudo();
          break;

        case 'd':
          Serial.println("Calibração: Motor 2 Direita");
          digitalWrite(pinoIN3_F, HIGH); digitalWrite(pinoIN4_F, LOW);
          delay(200); pararTudo();
          break;

        case 'a':
          Serial.println("Calibração: Motor 2 Esquerda");
          digitalWrite(pinoIN3_F, LOW); digitalWrite(pinoIN4_F, HIGH);
          delay(200); pararTudo();
          break;

        case 'z':
          sistemaCalibrado = true;
          sistemaPausado   = false;
          modoIA = false;
          movimentoIAAtivo = false;
          Serial.println("\n>>> INICIANDO CICLO AUTOMÁTICO DE FISIOTERAPIA <<<");
          mudarEstado(MOVENDO_FLEXAO);
          break;
      }
    } else {
      // Sistema ja calibrado: comandos de modo IA sob demanda.
      // 'z' continua sendo o ciclo automatico; 'm' entra no modo IA.
      if (comando == 'm') {
        modoIA = true;
        movimentoIAAtivo = false;
        pararTudo();
        Serial.println("\n>>> MODO IA: aguardando comandos 'l' (flexao) / 'e' (extensao) <<<");
      } else if (comando == 'l') {
        if (modoIA && !sistemaPausado) {
          movimentoIAAtivo = false;
          moverFlexao();
          movimentoIAAtivo = true;
          tempoInicioIA = millis();
          Serial.println(">> [IA] FLEXAO sob demanda");
        }
      } else if (comando == 'e') {
        if (modoIA && !sistemaPausado) {
          movimentoIAAtivo = false;
          moverExtensao();
          movimentoIAAtivo = true;
          tempoInicioIA = millis();
          Serial.println(">> [IA] EXTENSAO sob demanda");
        }
      } else if (comando == 'o') {
        movimentoIAAtivo = false;
        pararTudo();
        Serial.println(">> [IA] Parado");
      } else if (comando == 'z') {
        modoIA = false;
        movimentoIAAtivo = false;
        sistemaPausado = false;
        Serial.println("\n>>> INICIANDO CICLO AUTOMÁTICO DE FISIOTERAPIA <<<");
        mudarEstado(MOVENDO_FLEXAO);
      }
    }

    if (comando == 'p') {
      sistemaPausado = !sistemaPausado;
      if (sistemaPausado) {
        movimentoIAAtivo = false;
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
      movimentoIAAtivo = false;
      pararTudo();
      Serial.println("\n!!! RESET: Modo de calibração");
      exibirMenu();
    }
  }
}

void exibirMenu() {
  Serial.println("\n=============================================");
  Serial.println("    SISTEMA DE ÓRTESE ROBÓTICA - CIMATEC");
  Serial.println("=============================================");
  Serial.println("[ W / S ] Ajustar Motor 1 (Extensão)");
  Serial.println("[ D / A ] Ajustar Motor 2 (Flexão)");
  Serial.println("[   Z   ] Confirmar e Iniciar Rotina");
  Serial.println("[   M   ] Modo IA (comandos sob demanda)");
  Serial.println("[ L / E ] IA: flexao / extensao (modo IA)");
  Serial.println("[   O   ] IA: parar motores");
  Serial.println("[   P   ] Pausar / Retomar");
  Serial.println("[   R   ] Reiniciar Calibração");
  Serial.println("=============================================\n");
}