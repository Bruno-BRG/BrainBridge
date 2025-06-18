"""
Teste completo do ciclo LSL → Pré-processamento → Predição
Este script testa exatamente o que acontece no sistema real:
1. Recebe dados do LSL
2. Aplica o mesmo pré-processamento do notebook de treinamento
3. Faz predição a cada 400 amostras
4. Mostra resultados detalhados
"""

import numpy as np
import time
import logging
from collections import deque
from scipy import signal
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_window_exactly_like_notebook(data_window):
    """
    Pré-processamento EXATAMENTE igual ao notebook de treinamento
    Args:
        data_window: array (n_channels, n_samples) - 400 amostras
    
    Returns:
        normalized_data: array processado identicamente ao notebook
    """
    # Converter para numpy se necessário
    if not isinstance(data_window, np.ndarray):
        data_window = np.array(data_window)
    
    # Garantir formato (channels, samples) como no notebook
    if data_window.shape[0] != 16:
        data_window = data_window.T
    
    logger.info(f"📊 Pré-processamento NOTEBOOK: shape inicial = {data_window.shape}")
    
    # === FORMATO NOTEBOOK: Criar formato 3D (trials, channels, time) ===
    # O notebook usa formato (n_samples, n_channels, n_timepoints)
    data_3d = data_window[np.newaxis, :, :]  # Adicionar dimensão trial: (1, 16, 400)
    
    logger.info(f"📊 Shape 3D para notebook: {data_3d.shape}")
    
    # === NORMALIZAÇAO EXATA DO NOTEBOOK: ImprovedEEGNormalizer ===
    # method='robust_zscore', scope='channel'
    
    # 1. Handle outliers (robust method)
    Q1 = np.percentile(data_3d, 25, axis=(0, 2), keepdims=True)  # (1, channels, 1)
    Q3 = np.percentile(data_3d, 75, axis=(0, 2), keepdims=True)  # (1, channels, 1) 
    IQR = Q3 - Q1
    lower = Q1 - 3.0 * IQR  # outlier_threshold = 3.0
    upper = Q3 + 3.0 * IQR
    
    # Clip outliers
    data_clipped = np.clip(data_3d, lower, upper)
    logger.info(f"✅ Outliers tratados (robust clipping)")
    
    # 2. Robust Z-score normalization by channel (scope='channel')
    median = np.median(data_clipped, axis=(0, 2), keepdims=True)  # (1, channels, 1)
    q75, q25 = np.percentile(data_clipped, [75, 25], axis=(0, 2))
    iqr = (q75 - q25)[None, :, None] + 1e-8  # (1, channels, 1)
    
    # Normalize
    normalized_3d = (data_clipped - median) / iqr
    
    logger.info(f"✅ Normalização robust_zscore por canal aplicada")
    
    # === CONVERTER PARA FORMATO FINAL (channels, time) ===
    # Remover dimensão de trial para retornar (16, 400)
    normalized_data = normalized_3d.squeeze(0)  # (16, 400)
    
    # 3. Verificações finais IGUAIS AO NOTEBOOK
    logger.info(f"📊 Shape final: {normalized_data.shape}")
    logger.info(f"📊 Range final: [{np.min(normalized_data):.3f}, {np.max(normalized_data):.3f}]")
    logger.info(f"📊 Média por canal: {np.mean(normalized_data, axis=1)}")
    logger.info(f"📊 Std por canal: {np.std(normalized_data, axis=1)}")
    
    return normalized_data

class LSLToPredictionTester:
    """Testador do ciclo completo LSL → Predição"""
    
    def __init__(self, model_path="models/teste/eeginceptionerp_fold_1.pt"):
        self.model_path = model_path
        self.sample_rate = 125.0
        self.window_size = 400  # 400 amostras = ~3.2 segundos a 125 Hz
        self.n_channels = 16
        
        # Buffer para acumular amostras
        self.data_buffer = deque(maxlen=2000)  # Buffer maior para garantir
          # Controle de predições - fazer predição a cada N amostras
        self.prediction_interval = 400  # Fazer predição a cada 400 amostras novas
        self.last_prediction_sample = 0  # Controlar quando fazer próxima predição
        
        # Estatísticas
        self.samples_received = 0
        self.predictions_made = 0
        self.prediction_results = []
          # Carregar modelo
        self.load_model()
        
    def load_model(self):
        try:
            from inference_engine import EEGInferenceEngine
            self.inference_engine = EEGInferenceEngine(self.model_path)
            
            if self.inference_engine.is_loaded:
                logger.info(f"✅ Modelo carregado: {self.model_path}")
                logger.info(f"📊 Arquitetura: EEGInceptionERP")
                logger.info(f"📊 Canais: {self.inference_engine.n_chans}")
                logger.info(f"📊 Pontos temporais: {self.inference_engine.n_times}")
                logger.info(f"📊 Classes: {self.inference_engine.class_labels}")
                logger.info(f"🎯 Threshold de confiança: {self.inference_engine.confidence_threshold}")
                
                # Extrair estatísticas de normalização se disponíveis
                self.normalization_stats = self.inference_engine.normalizer_stats
                if self.normalization_stats:
                    logger.info(f"✅ Estatísticas de normalização carregadas do modelo")
                else:
                    logger.info(f"⚠️ Usando normalização Z-score padrão")
                
                return True
            else:
                logger.error(f"❌ Falha ao carregar modelo")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
        return False
    
    def add_sample(self, sample):
        """Adiciona nova amostra ao buffer"""
        if len(sample) >= self.n_channels:
            # Pegar apenas os primeiros 16 canais (EEG)
            eeg_sample = sample[:self.n_channels]
            self.data_buffer.append(eeg_sample)
            self.samples_received += 1
            
            # Log ocasional
            if self.samples_received % 50 == 0:
                sample_range = f"[{min(eeg_sample):.3f}, {max(eeg_sample):.3f}]"
                logger.info(f"📡 Sample #{self.samples_received}: buffer={len(self.data_buffer)}/{self.window_size}, range={sample_range}")
            
            # Verificar se deve fazer predição (a cada 400 amostras novas)
            samples_since_last_prediction = self.samples_received - self.last_prediction_sample
            
            if (len(self.data_buffer) >= self.window_size and 
                samples_since_last_prediction >= self.prediction_interval):
                
                logger.info(f"🎯 Fazendo predição - amostras desde última: {samples_since_last_prediction}")
                result = self.make_prediction()
                self.last_prediction_sample = self.samples_received
                return result
        
        return None
    
    def make_prediction(self):
        """Faz predição com as últimas 400 amostras"""
        if len(self.data_buffer) < self.window_size:
            logger.warning(f"⚠️ Buffer insuficiente: {len(self.data_buffer)}/{self.window_size}")
            return None
        
        self.predictions_made += 1
        logger.info(f"🔮 === PREDIÇÃO #{self.predictions_made} ===")
        
        # 1. Extrair janela de dados
        window_data = list(self.data_buffer)[-self.window_size:]  # Últimas 400 amostras
        data_array = np.array(window_data).T  # Transpor para (channels, samples)
        
        logger.info(f"📊 Janela extraída: {data_array.shape} (channels x samples)")
          # 2. Aplicar pré-processamento EXATAMENTE igual ao notebook
        try:
            processed_data = preprocess_window_exactly_like_notebook(data_array)
            logger.info(f"✅ Pré-processamento NOTEBOOK concluído")
        except Exception as e:
            logger.error(f"❌ Erro no pré-processamento: {e}")
            return None        # 3. Fazer predição usando o modelo COM THRESHOLD
        try:
            # Fazer predição básica
            predicted_class, confidence, probabilities = self.inference_engine.predict(processed_data)
            
            # Aplicar lógica de threshold
            is_confident = confidence >= self.inference_engine.confidence_threshold
            
            if is_confident:
                final_class = predicted_class
                class_label = self.inference_engine.class_labels[predicted_class]
            else:
                final_class = 2  # Classe "Incerto/Repouso"
                class_label = "Uncertain/Rest"
            
            result = {
                'prediction_number': self.predictions_made,
                'predicted_class': final_class,
                'class_label': class_label,
                'confidence': confidence,
                'is_confident': is_confident,
                'probabilities': probabilities,
                'samples_used': self.window_size,
                'timestamp': time.time()
            }
            
            self.prediction_results.append(result)
            
            logger.info(f"🎯 RESULTADO #{self.predictions_made}:")
            logger.info(f"   Classe: {final_class} ({class_label})")
            logger.info(f"   Confiança: {confidence:.3f} ({'CONFIANTE' if is_confident else 'INCERTA'})")
            logger.info(f"   Threshold: {self.inference_engine.confidence_threshold}")
            logger.info(f"   Probabilidades: [{probabilities[0]:.3f}, {probabilities[1]:.3f}]")
            logger.info(f"   Amostras usadas: {self.window_size}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            return None
    
    def run_lsl_test(self, duration_seconds=30):
        """Executa teste com dados LSL reais"""
        logger.info(f"🚀 === INICIANDO TESTE LSL → PREDIÇÃO ===")
        logger.info(f"⏱️ Duração: {duration_seconds} segundos")
        logger.info(f"📊 Janela de predição: {self.window_size} amostras (~{self.window_size/self.sample_rate:.1f}s)")
        
        # Importar e configurar LSL
        try:
            from lsl_streamer import LSLDataStreamer
            
            streamer = LSLDataStreamer()
            
            # Descobrir e conectar ao stream
            if not streamer.find_stream():
                logger.error("❌ Nenhum stream LSL encontrado")
                return False
            
            logger.info(f"✅ Conectado ao stream LSL")
            
            # Configurar callback
            def data_callback(sample, timestamp):
                result = self.add_sample(sample)
                if result:
                    # Nova predição disponível
                    pass
            
            streamer.set_data_callback(data_callback)
            
            # Iniciar gravação
            streamer.start_recording()
            logger.info(f"📡 Coletando dados LSL...")
            
            # Aguardar
            time.sleep(duration_seconds)
            
            # Parar
            streamer.stop_recording()
            logger.info(f"🛑 Teste finalizado")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no teste LSL: {e}")
            return False
    
    def run_simulated_test(self, duration_seconds=20):
        """Executa teste com dados simulados"""
        logger.info(f"🚀 === INICIANDO TESTE SIMULADO ===")
        logger.info(f"⏱️ Duração: {duration_seconds} segundos")
        logger.info(f"📊 Taxa de amostragem: {self.sample_rate} Hz")
        
        samples_per_second = int(self.sample_rate)
        total_samples = duration_seconds * samples_per_second
        
        logger.info(f"📊 Total de amostras esperadas: {total_samples}")
        
        for i in range(total_samples):
            # Simular dados EEG (ruído branco com algumas características)
            base_signal = np.random.randn(self.n_channels) * 0.1
            
            # Adicionar alguns padrões realistas
            time_point = i / self.sample_rate
            alpha_freq = 10  # 10 Hz alpha rhythm
            alpha_component = 0.05 * np.sin(2 * np.pi * alpha_freq * time_point)
            
            sample = base_signal + alpha_component
            
            # Processar amostra
            result = self.add_sample(sample)
            
            # Simular timing real
            if i % samples_per_second == 0:
                logger.info(f"⏱️ Segundo {i // samples_per_second}: {self.samples_received} amostras, {self.predictions_made} predições")
            
            time.sleep(1.0 / self.sample_rate)  # Simular taxa real
        
        logger.info(f"🛑 Teste simulado finalizado")
        return True
    
    def print_summary(self):
        """Imprime resumo dos resultados"""
        logger.info(f"\n🎉 === RESUMO DOS RESULTADOS ===")
        logger.info(f"📊 Amostras recebidas: {self.samples_received}")
        logger.info(f"📊 Predições realizadas: {self.predictions_made}")
        if self.prediction_results:
            # Análise das predições
            classes = [r['predicted_class'] for r in self.prediction_results]
            confidences = [r['confidence'] for r in self.prediction_results]
            is_confidents = [r.get('is_confident', True) for r in self.prediction_results]
            
            class_counts = np.bincount(classes, minlength=3)  # Agora temos 3 classes
            avg_confidence = np.mean(confidences)
            confident_count = sum(is_confidents)
            uncertain_count = len(is_confidents) - confident_count
            
            logger.info(f"📊 Distribuição de classes:")
            logger.info(f"   Mão Esquerda (0): {class_counts[0]} predições")
            logger.info(f"   Mão Direita (1): {class_counts[1]} predições")
            logger.info(f"   Incerto/Repouso (2): {class_counts[2]} predições")
            logger.info(f"📊 Predições confiantes: {confident_count}/{len(self.prediction_results)} ({confident_count/len(self.prediction_results)*100:.1f}%)")
            logger.info(f"📊 Predições incertas: {uncertain_count}/{len(self.prediction_results)} ({uncertain_count/len(self.prediction_results)*100:.1f}%)")
            logger.info(f"📊 Confiança média: {avg_confidence:.3f}")
            logger.info(f"📊 Confiança min/max: {min(confidences):.3f}/{max(confidences):.3f}")
            
            # Últimas 5 predições
            logger.info(f"📊 Últimas 5 predições:")
            for i, result in enumerate(self.prediction_results[-5:]):
                status = "✅ CONFIANTE" if result.get('is_confident', True) else "⚠️ INCERTA"
                logger.info(f"   {i+1}. {result['class_label']} (conf={result['confidence']:.3f}) {status}")
        
        logger.info(f"✅ Teste concluído com sucesso!")

def main():
    """Função principal"""
    print("🧠 === TESTE CICLO COMPLETO: LSL → PRÉ-PROCESSAMENTO → PREDIÇÃO ===\n")
    
    # Criar testador
    tester = LSLToPredictionTester()
    
    # Escolher tipo de teste
    print("Escolha o tipo de teste:")
    print("1. Teste com dados LSL reais")
    print("2. Teste com dados simulados")
    
    try:
        choice = input("Digite sua escolha (1 ou 2): ").strip()
        
        if choice == "1":
            success = tester.run_lsl_test(duration_seconds=30)
        elif choice == "2":
            success = tester.run_simulated_test(duration_seconds=20)
        else:
            print("❌ Escolha inválida")
            return
        
        if success:
            tester.print_summary()
        else:
            print("❌ Teste falhou")
            
    except KeyboardInterrupt:
        print("\n🛑 Teste interrompido pelo usuário")
        tester.print_summary()
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")

if __name__ == "__main__":
    main()
