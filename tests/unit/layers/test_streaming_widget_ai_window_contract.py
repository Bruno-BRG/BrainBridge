import ast
from pathlib import Path


STREAMING_WIDGET_PATH = (
    Path(__file__).resolve().parents[3]
    / "brainbridge_v2"
    / "presentation"
    / "gui"
    / "widgets"
    / "streaming.py"
)


def _method_node(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"Metodo {method_name} nao encontrado.")


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return None


def _streaming_widget_class() -> ast.ClassDef:
    tree = ast.parse(STREAMING_WIDGET_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "StreamingWidget":
            return node
    raise AssertionError("Classe StreamingWidget nao encontrada.")


def test_ai_window_resets_samples_collected_before_unity_marker():
    widget = _streaming_widget_class()
    method = _method_node(widget, "_start_ai_prediction_window")

    calls = [_call_name(node) for node in ast.walk(method)]

    assert "start_window" in calls
    assert "_sync_ai_prediction_state" in calls


def test_all_game_cues_open_window_in_marker_handler():
    widget = _streaming_widget_class()

    for method_name in ("send_next_random_signal", "game_random_action"):
        method = _method_node(widget, method_name)
        ordered_calls = [
            _call_name(node.value)
            for node in ast.walk(method)
            if isinstance(node, ast.Expr)
        ]

        assert "add_marker" in ordered_calls
        assert "_start_ai_prediction_window" not in ordered_calls
    marker = _method_node(widget, "add_marker")
    assert "_start_ai_prediction_window" in [_call_name(n) for n in ast.walk(marker)]


def test_streaming_widget_records_pipeline_events_for_critical_flow():
    widget = _streaming_widget_class()
    source = STREAMING_WIDGET_PATH.read_text(encoding="utf-8")

    for event_name in (
        "TASK_SENT",
        "AI_WINDOW_OPENED",
        "SAMPLE_250_READY",
        "WINDOW_REJECTED",
        "PREDICTION_DONE",
        "UNITY_COMMAND_SENT",
        "UNITY_RESPONSE",
    ):
        assert event_name in source


def test_streaming_widget_exposes_optional_developer_telemetry_mode():
    widget = _streaming_widget_class()
    source = STREAMING_WIDGET_PATH.read_text(encoding="utf-8")

    assert "DeveloperSettingsDialog" in source
    assert "open_developer_settings" in source
    assert "set_developer_mode" in source
    assert "PipelineTelemetry(enabled=False)" in source
    assert "pipeline_telemetry.set_enabled" in source


def test_streaming_widget_uses_quality_validator_and_command_mapper():
    source = STREAMING_WIDGET_PATH.read_text(encoding="utf-8")

    assert "EEGWindowQualityValidator" in source
    assert "UnityCommandMapper.from_prediction" in source
