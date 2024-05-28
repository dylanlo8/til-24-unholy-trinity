from faster_whisper import WhisperModel
from pydub import AudioSegment  
import os
import inflect
import noisereduce as nr

class ASRManager:
    def __init__(self):
        # initialize the model here
        # load model and processor
        model_size = "medium"
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def transcribe(self, audio_bytes: bytes) -> str:
        audio_segment = AudioSegment(
            data=audio_bytes,
            sample_width=2, 
            frame_rate=16000,
            channels=1 
        )
        
        '''
        # Denoise the audio
        audio_array = audio_segment.get_array_of_samples()
        denoised_audio = nr.reduce_noise(audio_array, audio_array)

        # Save denoised audio to a temporary WAV file
        denoised_audio_segment = AudioSegment(
            data=denoised_audio,
            sample_width=2,
            frame_rate=16000,
            channels=1
        )
        denoised_audio_segment.export('denoised_output.wav', format='wav')

        # Perform ASR transcription
        segments, info = self.model.transcribe('denoised_output.wav', beam_size=5)

        # Clean up temporary WAV file
        os.remove('denoised_output.wav')

        segments = list(segments)
        return self.convert_numbers_to_text(segments[0].text.split())
        
        ->if want to use, then can remove (not tested yet)
        '''
        
        
        audio_segment.export('output.wav', format='wav')

        # Perform ASR transcription
        segments, info = self.model.transcribe('output.wav', beam_size=5) 

        # Clean up wav file
        os.remove('output.wav')

        segments = list(segments)
        
        prediction_texts = [self.process_text(segment.text) for segment in segments]

        # Join all extracted texts into a single string
        combined_text = ' '.join(prediction_texts)
    
     
        return combined_text.strip()

    
    def number_to_text(self, n):
        num_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'niner'
        }
        return ' '.join(num_map.get(digit, digit) for digit in str(n))

    def replace_dashes(self, text):
        return text.replace('-', ' ')

    def process_text(self, text):
        text = self.replace_dashes(text)
        words = text.split()
        processed_words = []

        for word in words:
            if word.isdigit():
                # If the entire word is digits, convert it directly
                processed_word = self.number_to_text(word)
            else:
                # If the word contains a mix of digits and letters, process it carefully
                processed_parts = []
                current_part = ''

                for char in word:
                    if char.isdigit():
                        if current_part and not current_part[-1].isdigit():
                            # If switching from letters to digits, store the current part and reset
                            processed_parts.append(current_part)
                            current_part = char
                        else:
                            # Continue appending digits
                            current_part += char
                    else:
                        if current_part and current_part[-1].isdigit():
                            # If switching from digits to letters, convert the digit part and reset
                            processed_parts.append(self.number_to_text(current_part))
                            current_part = char
                        else:
                            # Continue appending letters
                            current_part += char

                # Append the last part
                if current_part:
                    if current_part.isdigit():
                        processed_parts.append(self.number_to_text(current_part))
                    else:
                        processed_parts.append(current_part)

                processed_word = ' '.join(processed_parts)

            processed_words.append(processed_word)

        return ' '.join(processed_words)
