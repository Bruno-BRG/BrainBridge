"""
LSL (Lab Streaming Layer) module for real-time EEG data acquisition
Handles streaming data from OpenBCI and other EEG devices
"""
import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_streams, StreamInfo, StreamOutlet, resolve_byprop
import time
import threading
from datetime import datetime
from typing import Optional, Callable, List, Dict
import logging
from pathlib import Path
import queue

logger = logging.getLogger(__name__)

class LSLDataStreamer:
    """LSL Data Streamer for EEG acquisition"""
    
    def __init__(self, sample_rate: float = 125.0, n_channels: int = 16):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.inlet: Optional[StreamInlet] = None
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.data_buffer = queue.Queue()
        self.annotations = []
        self.recording_data = []
        self.start_time = None
        self.available_streams = []
        
        # Callbacks
        self.data_callback: Optional[Callable] = None
        self.annotation_callback: Optional[Callable] = None
        
    def discover_streams(self, timeout: float = 5.0) -> List[Dict]:
        """Discover all available LSL streams on the network"""
        try:
            logger.info("Discovering LSL streams...")
            
            # Try different methods to find streams
            all_streams = []
            
            # Method 1: Find by type (correct pylsl syntax)
            for stream_type in ['EEG', 'EMG', 'ECG', 'ExG']:
                try:
                    streams = resolve_byprop('type', stream_type, timeout=timeout/4)
                    all_streams.extend(streams)
                    if streams:
                        logger.info(f"Found {len(streams)} {stream_type} streams")
                except Exception as e:
                    logger.debug(f"No {stream_type} streams found: {e}")
            
            # Method 2: Find all streams without filter (pylsl correct syntax)
            try:
                general_streams = resolve_streams(wait_time=timeout/2)
                all_streams.extend(general_streams)
                if general_streams:
                    logger.info(f"Found {len(general_streams)} general streams")
            except Exception as e:
                logger.debug(f"General stream discovery failed: {e}")
                
            # Method 3: Try different common stream names
            common_names = ['OpenBCI', 'EEG', 'BrainVision', 'g.Nautilus', 'ActiChamp']
            for name in common_names:
                try:
                    streams = resolve_byprop('name', name, timeout=timeout/6)
                    all_streams.extend(streams)
                    if streams:
                        logger.info(f"Found {len(streams)} streams with name {name}")
                except Exception as e:
                    logger.debug(f"No streams found with name {name}: {e}")
            
            # Remove duplicates based on source_id
            unique_streams = {}
            for stream in all_streams:
                source_id = stream.source_id()
                if source_id not in unique_streams:
                    unique_streams[source_id] = stream
            
            # Format stream information
            self.available_streams = []
            for stream in unique_streams.values():
                stream_info = {
                    'name': stream.name(),
                    'type': stream.type(),
                    'source_id': stream.source_id(),
                    'hostname': stream.hostname(),
                    'channel_count': stream.channel_count(),
                    'nominal_srate': stream.nominal_srate(),
                    'channel_format': stream.channel_format(),
                    'stream_obj': stream
                }
                self.available_streams.append(stream_info)
                
            logger.info(f"Found {len(self.available_streams)} unique streams")
            for stream in self.available_streams:
                logger.info(f"Stream: {stream['name']} ({stream['type']}) - {stream['channel_count']} channels @ {stream['nominal_srate']} Hz")
            
            return self.available_streams
            
        except Exception as e:
            logger.error(f"Error discovering streams: {e}")
            return []
    
    def connect_to_stream(self, stream_info: Dict) -> bool:
        """Connect to a specific stream"""
        try:
            logger.info(f"Connecting to stream: {stream_info['name']}")
            self.inlet = StreamInlet(stream_info['stream_obj'])
            
            # Update local info based on actual stream
            info = self.inlet.info()
            self.n_channels = info.channel_count()
            self.sample_rate = info.nominal_srate()
            
            logger.info(f"Connected to {stream_info['name']} - {self.n_channels} channels @ {self.sample_rate} Hz")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to stream: {e}")
            return False
        
    def find_stream(self, stream_type: str = "EEG", timeout: float = 5.0) -> bool:
        """Find and connect to LSL stream (legacy method)"""
        streams = self.discover_streams(timeout)
        
        if not streams:
            logger.warning("No LSL streams found. Please check:")
            logger.warning("1. Your EEG device is connected and streaming")
            logger.warning("2. The LSL software is running")
            logger.warning("3. Firewall is not blocking LSL ports")
            return False
        
        # Try to find EEG stream first
        for stream in streams:
            if stream['type'].upper() == stream_type.upper():
                return self.connect_to_stream(stream)
        
        # If no EEG stream found, connect to first available
        if streams:
            logger.info(f"No {stream_type} stream found, connecting to first available stream")
            return self.connect_to_stream(streams[0])
        
        return False
    
    def start_recording(self, output_file: Optional[str] = None):
        """Start recording data from stream"""
        if self.inlet is None:
            raise ValueError("No stream connected. Call find_stream() first.")
        
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        self.is_recording = True
        self.recording_data = []
        self.annotations = []
        self.start_time = time.time()
        
        # Setup output file
        if output_file:
            self.output_file = output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"eeg_recording_{timestamp}.csv"
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        logger.info(f"Started recording to {self.output_file}")
    
    def stop_recording(self) -> str:
        """Stop recording and save data"""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return ""
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # Save data to CSV
        if self.recording_data:
            self._save_to_csv()
            logger.info(f"Recording saved to {self.output_file}")
            return self.output_file
        else:
            logger.warning("No data recorded")
            return ""
    
    def _recording_loop(self):
        """Main recording loop running in separate thread"""
        sample_index = 0
        last_log_time = time.time()
        
        logger.info("🔄 Iniciando loop de gravação LSL...")
        
        while self.is_recording:
            try:
                # Pull sample from stream
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    sample_index += 1
                    current_time = time.time() - self.start_time
                    
                    # Log detalhado dos dados recebidos (ocasionalmente)
                    if sample_index % 50 == 0 or time.time() - last_log_time > 5.0:
                        sample_summary = f"shape={len(sample)}, range=[{min(sample):.3f}, {max(sample):.3f}]"
                        logger.info(f"📡 LSL Sample #{sample_index}: {sample_summary}, timestamp={timestamp:.3f}")
                        last_log_time = time.time()
                    
                    # Create data row
                    row = {
                        'Sample Index': sample_index,
                        'Timestamp': current_time,
                        'LSL_Timestamp': timestamp,
                        'Annotations': ''
                    }
                    
                    # Add EEG channels
                    for i in range(min(len(sample), self.n_channels)):
                        row[f'EXG Channel {i}'] = sample[i]
                    
                    # Add to buffer for real-time processing
                    if self.data_callback:
                        try:
                            # Log antes de chamar o callback
                            if sample_index % 50 == 0:
                                logger.info(f"🔄 Enviando sample #{sample_index} para callback (n_channels={len(sample)})")
                            
                            self.data_callback(sample, timestamp)
                            
                            # Log após chamar o callback
                            if sample_index % 50 == 0:
                                logger.info(f"✅ Callback executado com sucesso para sample #{sample_index}")
                        except Exception as callback_error:
                            logger.error(f"❌ Erro no callback para sample #{sample_index}: {callback_error}")
                    else:
                        if sample_index % 100 == 0:
                            logger.warning(f"⚠️ Nenhum callback configurado - sample #{sample_index} perdido")
                    
                    # Store for file output
                    self.recording_data.append(row)
                else:
                    # Log quando não conseguimos puxar amostra
                    if sample_index % 200 == 0:
                        logger.warning(f"⚠️ Nenhuma amostra recebida do LSL (timeout)")
                    
            except Exception as e:
                logger.error(f"❌ Erro no loop de gravação: {e}")
                time.sleep(0.1)
    
    def add_annotation(self, annotation: str, hand: str = ""):
        """Add annotation/marker to current recording"""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return
        
        current_time = time.time() - self.start_time if self.start_time else 0
        sample_index = len(self.recording_data)
        
        annotation_data = {
            'timestamp': current_time,
            'sample_index': sample_index,
            'annotation': annotation,
            'hand': hand
        }
        
        self.annotations.append(annotation_data)
        
        # Update the current sample with annotation
        if self.recording_data:
            self.recording_data[-1]['Annotations'] = annotation
        
        if self.annotation_callback:
            self.annotation_callback(annotation_data)
        
        logger.info(f"Added annotation: {annotation} at sample {sample_index}")
    
    def _save_to_csv(self):
        """Save recorded data to CSV file"""
        if not self.recording_data:
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.recording_data)
        
        # Create header with metadata (OpenBCI format)
        header_lines = [
            "%OpenBCI Raw EXG Data",
            f"%Number of channels = {self.n_channels}",
            f"%Sample Rate = {self.sample_rate} Hz",
            "%Board = LSL_Stream",
            ""
        ]
        
        # Write header and data
        with open(self.output_file, 'w', newline='') as f:
            # Write header
            for line in header_lines[:-1]:
                f.write(line + '\n')
            
            # Write CSV data
            df.to_csv(f, index=False)
        
        # Save annotations separately
        if self.annotations:
            annotations_file = self.output_file.replace('.csv', '_annotations.json')
            import json
            with open(annotations_file, 'w') as f:
                json.dump(self.annotations, f, indent=2)
    
    def get_stream_info(self) -> Dict:
        """Get current stream information"""
        if self.inlet is None:
            return {}
        
        info = self.inlet.info()
        return {
            'name': info.name(),
            'type': info.type(),
            'channel_count': info.channel_count(),
            'sample_rate': info.nominal_srate(),
            'source_id': info.source_id()
        }
    
    def set_data_callback(self, callback: Callable):
        """Set callback function for real-time data processing"""
        self.data_callback = callback
    
    def set_annotation_callback(self, callback: Callable):
        """Set callback function for annotation events"""
        self.annotation_callback = callback
    
    def is_connected(self) -> bool:
        """Check if connected to stream"""
        return self.inlet is not None
    
    def disconnect(self):
        """Disconnect from stream"""
        if self.is_recording:
            self.stop_recording()
        
        self.inlet = None
        logger.info("Disconnected from stream")


class HandMovementAnnotator:
    """Helper class for hand movement annotations during recording"""
    
    def __init__(self, streamer: LSLDataStreamer, samples_per_annotation: int = 400):
        self.streamer = streamer
        self.samples_per_annotation = samples_per_annotation
        self.current_annotation = None
        self.annotation_start_sample = 0
        self.sample_count = 0
        
        # Set up data callback to count samples
        self.streamer.set_data_callback(self._on_data_sample)
    
    def _on_data_sample(self, sample, timestamp):
        """Called for each new data sample"""
        self.sample_count += 1
        
        # Check if current annotation period has ended
        if (self.current_annotation and 
            self.sample_count - self.annotation_start_sample >= self.samples_per_annotation):
            self._end_current_annotation()
    
    def start_left_hand_annotation(self):
        """Start left hand movement annotation"""
        self._start_annotation("T1", "left")
    
    def start_right_hand_annotation(self):
        """Start right hand movement annotation"""
        self._start_annotation("T2", "right")
    
    def _start_annotation(self, annotation: str, hand: str):
        """Start a new annotation period"""
        if self.current_annotation:
            self._end_current_annotation()
        
        self.streamer.add_annotation(annotation, hand)
        self.current_annotation = annotation
        self.annotation_start_sample = self.sample_count
        
        logger.info(f"Started {hand} hand annotation ({annotation}) for {self.samples_per_annotation} samples")
    
    def _end_current_annotation(self):
        """End current annotation period"""
        if self.current_annotation:
            self.streamer.add_annotation("T0", "rest")
            logger.info(f"Ended annotation {self.current_annotation}")
            self.current_annotation = None
    
    def force_end_annotation(self):
        """Manually end current annotation"""
        self._end_current_annotation()
