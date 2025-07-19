"""
Audio Processing Module for Real-time Speech Analysis
Handles audio capture, speaker diarization, and sentiment analysis
"""

import numpy as np
import threading
import time
from typing import Dict, List, Any, Optional
from collections import deque
import wave
import io

from utils.base_processor import BaseProcessor
from utils.logger import logger

# Check for optional dependencies
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Audio processing will be disabled.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: Librosa not available. Advanced audio analysis disabled.")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Warning: SpeechRecognition not available. Speech analysis disabled.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Text sentiment analysis disabled.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VaderSentiment not available. Sentiment analysis disabled.")

class AudioProcessor(BaseProcessor):
    """Real-time audio processing for engagement analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AudioProcessor", config)

        # Check if audio processing is available
        self.audio_available = PYAUDIO_AVAILABLE

        # Audio configuration
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)
        self.chunk_size = config.get('chunk_size', 1024)
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None

        # Audio processing components
        self.audio = None
        self.stream = None
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=self.sample_rate * 10)  # 10 seconds buffer
        self.speech_segments = []
        self.silence_threshold = config.get('silence_threshold', 0.01)
        
        # Speech recognition
        self.min_speech_duration = config.get('min_speech_duration', 1.0)
        self.max_speech_duration = config.get('max_speech_duration', 30.0)
        self.speech_queue = deque(maxlen=50)
        
        # Sentiment analysis
        self.sentiment_history = deque(maxlen=100)
        self.current_sentiment = {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Speaker analysis
        self.speaker_activity = {}
        self.total_speech_time = 0.0
        self.silence_time = 0.0
        
        # Audio capture thread
        self.capture_thread = None
        self.is_capturing = False
    
    def initialize(self) -> bool:
        """Initialize audio processing components"""
        try:
            if not self.audio_available:
                logger.warning("Audio processing not available - PyAudio not installed")
                return True  # Return True to continue without audio

            logger.info("Initializing audio processing...")

            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()

            # Find default input device
            default_device = self.audio.get_default_input_device_info()
            logger.info(f"Using audio device: {default_device['name']}")

            # Setup audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            # Configure speech recognizer
            if self.recognizer:
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            
            logger.info("Audio processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")
            return False
    
    def start(self) -> bool:
        """Start audio processing"""
        if not super().start():
            return False
        
        try:
            # Start audio capture
            self.is_capturing = True
            self.stream.start_stream()
            
            # Start speech processing thread
            self.capture_thread = threading.Thread(target=self._speech_processing_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info("Audio capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def stop(self):
        """Stop audio processing"""
        self.is_capturing = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        super().stop()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            return (in_data, pyaudio.paAbort)
    
    def process_data(self, data: Any = None) -> Dict[str, Any]:
        """Process audio data for engagement analysis"""
        try:
            if len(self.audio_buffer) == 0:
                return self._empty_result()
            
            # Get recent audio data
            audio_data = np.array(list(self.audio_buffer))
            
            # Analyze audio features
            audio_features = self._analyze_audio_features(audio_data)
            
            # Detect speech activity
            speech_activity = self._detect_speech_activity(audio_data)
            
            # Get recent speech recognition results
            recent_speech = self._get_recent_speech()
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_sentiment(recent_speech)
            
            # Calculate engagement metrics
            engagement_metrics = self._calculate_audio_engagement()
            
            result = {
                'audio_features': audio_features,
                'speech_activity': speech_activity,
                'recent_speech': recent_speech,
                'sentiment_analysis': sentiment_analysis,
                'engagement_metrics': engagement_metrics,
                'speaker_stats': self._get_speaker_stats()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return self._empty_result(error=str(e))
    
    def _analyze_audio_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze audio features for engagement"""
        try:
            # Convert to float
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Calculate basic features
            rms_energy = np.sqrt(np.mean(audio_float ** 2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_float)[0])
            
            # Spectral features
            if len(audio_float) > 512:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_float, sr=self.sample_rate)[0])
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_float, sr=self.sample_rate)[0])
                mfcc = np.mean(librosa.feature.mfcc(y=audio_float, sr=self.sample_rate, n_mfcc=13), axis=1)
            else:
                spectral_centroid = 0.0
                spectral_rolloff = 0.0
                mfcc = np.zeros(13)
            
            return {
                'rms_energy': float(rms_energy),
                'zero_crossing_rate': float(zero_crossing_rate),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'mfcc': mfcc.tolist(),
                'is_speech': rms_energy > self.silence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio features: {e}")
            return {'error': str(e)}
    
    def _detect_speech_activity(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect speech activity and speaker changes"""
        try:
            # Calculate energy
            energy = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            
            # Determine if speech is present
            is_speech = energy > self.silence_threshold
            
            # Update speech/silence timing
            current_time = time.time()
            if is_speech:
                self.total_speech_time += 0.1  # Approximate chunk duration
            else:
                self.silence_time += 0.1
            
            # Simple speaker activity detection (based on energy levels)
            if is_speech:
                energy_level = 'high' if energy > 0.05 else 'medium' if energy > 0.02 else 'low'
                speaker_id = f"speaker_{energy_level}"  # Simplified speaker identification
                
                if speaker_id not in self.speaker_activity:
                    self.speaker_activity[speaker_id] = {'total_time': 0.0, 'last_active': current_time}
                
                self.speaker_activity[speaker_id]['total_time'] += 0.1
                self.speaker_activity[speaker_id]['last_active'] = current_time
            
            return {
                'is_speech': is_speech,
                'energy_level': float(energy),
                'speech_ratio': self.total_speech_time / (self.total_speech_time + self.silence_time) if (self.total_speech_time + self.silence_time) > 0 else 0.0,
                'active_speakers': len([s for s in self.speaker_activity.values() if current_time - s['last_active'] < 5.0])
            }
            
        except Exception as e:
            logger.error(f"Error detecting speech activity: {e}")
            return {'error': str(e)}
    
    def _speech_processing_loop(self):
        """Background thread for speech recognition"""
        while self.is_capturing:
            try:
                if len(self.audio_buffer) >= self.sample_rate * 2:  # 2 seconds of audio
                    # Get audio segment
                    audio_segment = np.array(list(self.audio_buffer)[-self.sample_rate * 2:])
                    
                    # Check if segment contains speech
                    energy = np.sqrt(np.mean(audio_segment.astype(np.float32) ** 2))
                    
                    if energy > self.silence_threshold:
                        # Convert to audio format for speech recognition
                        audio_bytes = self._numpy_to_wav_bytes(audio_segment)
                        
                        try:
                            # Recognize speech
                            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                                audio_data = self.recognizer.record(source)
                                text = self.recognizer.recognize_google(audio_data, language='en-US')
                                
                                if text.strip():
                                    self.speech_queue.append({
                                        'timestamp': time.time(),
                                        'text': text,
                                        'confidence': 0.8  # Google API doesn't provide confidence
                                    })
                                    
                                    logger.info(f"Speech recognized: {text}")
                        
                        except sr.UnknownValueError:
                            pass  # No speech recognized
                        except sr.RequestError as e:
                            logger.error(f"Speech recognition error: {e}")
                
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in speech processing loop: {e}")
                time.sleep(1.0)
    
    def _numpy_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes"""
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.astype(np.int16).tobytes())
        
        buffer.seek(0)
        return buffer.read()
    
    def _get_recent_speech(self) -> List[Dict[str, Any]]:
        """Get recent speech recognition results"""
        current_time = time.time()
        return [speech for speech in self.speech_queue 
                if current_time - speech['timestamp'] < 30.0]  # Last 30 seconds
    
    def _analyze_sentiment(self, speech_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of recent speech"""
        if not speech_data:
            return self.current_sentiment.copy()
        
        # Combine recent text
        recent_text = ' '.join([speech['text'] for speech in speech_data[-5:]])  # Last 5 utterances
        
        if not recent_text.strip():
            return self.current_sentiment.copy()
        
        try:
            # VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(recent_text)
            
            # TextBlob sentiment analysis
            blob = TextBlob(recent_text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Combine scores
            combined_sentiment = {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'polarity': textblob_polarity,
                'subjectivity': textblob_subjectivity,
                'text_analyzed': recent_text[:100] + '...' if len(recent_text) > 100 else recent_text
            }
            
            # Update current sentiment (moving average)
            self.current_sentiment = {
                'compound': 0.7 * self.current_sentiment['compound'] + 0.3 * combined_sentiment['compound'],
                'positive': 0.7 * self.current_sentiment['positive'] + 0.3 * combined_sentiment['positive'],
                'negative': 0.7 * self.current_sentiment['negative'] + 0.3 * combined_sentiment['negative'],
                'neutral': 0.7 * self.current_sentiment['neutral'] + 0.3 * combined_sentiment['neutral']
            }
            
            # Add to history
            self.sentiment_history.append({
                'timestamp': time.time(),
                'sentiment': combined_sentiment
            })
            
            return combined_sentiment
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self.current_sentiment.copy()
    
    def _calculate_audio_engagement(self) -> Dict[str, Any]:
        """Calculate audio-based engagement metrics"""
        try:
            current_time = time.time()
            
            # Speech activity engagement
            total_time = self.total_speech_time + self.silence_time
            speech_ratio = self.total_speech_time / total_time if total_time > 0 else 0.0
            
            # Sentiment engagement (positive sentiment indicates engagement)
            sentiment_score = max(0, self.current_sentiment['compound'])  # 0 to 1
            
            # Speaker diversity (more speakers = more engagement)
            active_speakers = len([s for s in self.speaker_activity.values() 
                                 if current_time - s['last_active'] < 10.0])
            speaker_diversity = min(1.0, active_speakers / 5.0)  # Normalize to 0-1
            
            # Recent speech frequency
            recent_speech = self._get_recent_speech()
            speech_frequency = len(recent_speech) / 30.0  # Utterances per second (last 30s)
            speech_frequency_score = min(1.0, speech_frequency * 10)  # Normalize
            
            # Combined audio engagement score
            audio_engagement = (
                speech_ratio * 0.3 +
                sentiment_score * 0.3 +
                speaker_diversity * 0.2 +
                speech_frequency_score * 0.2
            )
            
            return {
                'audio_engagement_score': audio_engagement,
                'speech_ratio': speech_ratio,
                'sentiment_score': sentiment_score,
                'speaker_diversity': speaker_diversity,
                'speech_frequency': speech_frequency,
                'engagement_level': 'high' if audio_engagement > 0.7 else 'medium' if audio_engagement > 0.4 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error calculating audio engagement: {e}")
            return {'error': str(e)}
    
    def _get_speaker_stats(self) -> Dict[str, Any]:
        """Get speaker statistics"""
        current_time = time.time()
        
        active_speakers = {
            speaker_id: {
                'total_time': data['total_time'],
                'last_active': data['last_active'],
                'is_currently_active': current_time - data['last_active'] < 2.0
            }
            for speaker_id, data in self.speaker_activity.items()
        }
        
        return {
            'total_speakers': len(self.speaker_activity),
            'active_speakers': len([s for s in active_speakers.values() if s['is_currently_active']]),
            'speaker_details': active_speakers,
            'total_speech_time': self.total_speech_time,
            'total_silence_time': self.silence_time
        }
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'audio_features': {},
            'speech_activity': {'is_speech': False, 'energy_level': 0.0},
            'recent_speech': [],
            'sentiment_analysis': self.current_sentiment.copy(),
            'engagement_metrics': {'audio_engagement_score': 0.0},
            'speaker_stats': {'total_speakers': 0, 'active_speakers': 0}
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def cleanup(self):
        """Cleanup audio resources"""
        self.is_capturing = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        logger.info("Audio processor cleaned up")
