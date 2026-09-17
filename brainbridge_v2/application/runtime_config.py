"""
Runtime defaults for the EEG game pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    sample_rate_hz: int = 125
    window_size: int = 250
    channels: int = 16
    ai_window_duration_ms: int = 2000
    game_action_interval_ms: int = 10000
    eeg_max_abs_amplitude: float = 5000.0
    eeg_min_channel_std: float = 1e-6
    tensorflow_warmup_enabled: bool = True
    # Modo Livre (sem VR obrigatorio): janela continua com overlap.
    free_stride: int = 125  # 50% overlap @125Hz -> ~1 predicao/s
    free_min_confidence: float = 0.0  # exibe tudo; filtra so no envio
    free_send_min_confidence: float = 0.6  # so aciona VR/ortese acima disso
    free_send_cooldown_ms: int = 3000  # mesma ordem do trigger da ortese


DEFAULT_RUNTIME_CONFIG = RuntimeConfig()


# ---------------------------------------------------------------------------
# Overrides mutaveis de sessao (calibracao obrigatoria + RL online).
# Lidos via get_runtime(); a UI de configuracoes escreve via set_runtime().
# Escopo = processo atual (sem persistencia; documentado).
# ---------------------------------------------------------------------------
_CALIBRATION_DEFAULTS = {
    # Calibracao obrigatoria (1x por paciente)
    "calib_trials_required": 10,  # trials T1/T2 minimos p/ treinar (padrao clinico)
    "calib_epochs": 10,  # ablacao S012: 10ep estavel; 15ep overfita (val 0.46)
    "calib_lr": 5e-5,  # rede toda (ablacao: head-only nao move)
    "calib_freeze_backbone": False,
    # RL online (feedback humano durante a inferencia)
    "rl_enabled": False,
    "rl_batch_k": 5,  # aplica sozinho a cada K feedbacks rotulados
    "rl_epochs": 3,
    "rl_lr": 5e-5,
    "rl_max_updates": 20,  # trava de seguranca por sessao
    "rl_mistake_weight": 3.0,  # erro pesa mais que acerto
    "rl_augment": True,  # 1 replica com ruido leve por update (estabilidade)
    "rl_buffer_max": 200,
}

_RUNTIME_OVERRIDES: dict = {}


def get_runtime(name: str, default=None):
    """Le configuracao de sessao (override ou default de calibracao/RL)."""
    if name in _RUNTIME_OVERRIDES:
        return _RUNTIME_OVERRIDES[name]
    if name in _CALIBRATION_DEFAULTS:
        return _CALIBRATION_DEFAULTS[name]
    return default


def set_runtime(name: str, value) -> None:
    """Define override de sessao. Chave desconhecida levanta ValueError."""
    if name not in _CALIBRATION_DEFAULTS:
        raise ValueError(f"Configuracao desconhecida: {name}")
    _RUNTIME_OVERRIDES[name] = value


def reset_runtime() -> None:
    """Limpa overrides (uso em testes)."""
    _RUNTIME_OVERRIDES.clear()
