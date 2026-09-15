from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from brainbridge_v2.presentation.gui.widgets.streaming import StreamingWidget


def test_visual_filter_cannot_modify_raw_inference_or_logger_input():
    raw = np.arange(16, dtype=float) + 10000
    expected = raw.copy()
    def visual_filter(sample):
        sample[:] = -1
        return sample
    widget = SimpleNamespace(
        eeg_connection_phase='connected',
        plot_filter=SimpleNamespace(apply_realtime_filter=visual_filter),
        plot_widget=Mock(), pipeline_telemetry=Mock(),
        _is_game_mode=lambda: True, _sync_ai_prediction_state=Mock(),
        _ea_feed_sample=lambda data: False,
        game_inference=Mock(), is_recording=True, csv_logger=Mock(),
        pending_marker='T1', channels=16,
    )
    StreamingWidget.on_data_received(widget, raw)
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(widget.game_inference.add_sample.call_args.args[0], expected)
    np.testing.assert_array_equal(widget.csv_logger.log_sample.call_args.args[0], expected)
    np.testing.assert_array_equal(widget.plot_widget.add_data.call_args.args[0], np.full(16, -1))
