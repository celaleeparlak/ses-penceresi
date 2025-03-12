###                                                                    ###
###                    TÜBİTAK Ses Penceresi Projesi                   ###
###      Erdemli Borsa İstanbul Fen Lisesi için geliştirilmiştir.      ###
###                                                                    ###
###                      Yazar: Celal Efe PARLAK                       ###
###                     celalefeparlak@outlook.com                     ###
###                                                                    ###


import time
import pygame
import pyaudio
import numpy as np

from pygame import gfxdraw
from collections import deque


class AudioVisualizer:
    def __init__(self):
        self.RATE = 44100
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1

        self.screen_width = 1200
        self.screen_height = 800
        self.bg_color = (0, 0, 0)
        self.fg_color = (255, 255, 255)
        self.max_history = 100

        self.ui_opacity = 180
        self.ui_accent_color = (220, 220, 220)
        self.ui_bg_color = (30, 30, 30)
        self.ui_width = 300
        self.toggle_cooldown = 0

        self.show_ui = True
        self.ui_animation_active = False
        self.ui_animation_progress = 1.0
        self.ui_animation_speed = 0.05
        self.ui_animation_target = 1.0

        self.audio_data     = np.zeros(self.CHUNK)
        self.volume_history = deque(maxlen = self.max_history)
        self.freq_history   = deque(maxlen = self.max_history)
        self.bass_history   = deque(maxlen = self.max_history)
        self.mid_history    = deque(maxlen = self.max_history)
        self.high_history   = deque(maxlen = self.max_history)

        self.running = False
        self.p = pyaudio.PyAudio()
        self.initialize_pygame()


    def initialize_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Ses Penceresi")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 20, bold = True)


    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype = np.int16)
        self.audio_data = audio_data / 32768.0

        self.process_audio_features()

        return (in_data, pyaudio.paContinue)


    def process_audio_features(self):
        volume = np.abs(self.audio_data).mean()
        self.volume_history.append(volume)

        fft_data = np.abs(np.fft.rfft(self.audio_data))
        fft_freq = np.fft.rfftfreq(len(self.audio_data), 1.0/self.RATE)

        bass_idx = np.where((fft_freq >= 20) & (fft_freq <= 250))[0]
        mid_idx = np.where((fft_freq > 250) & (fft_freq <= 2000))[0]
        high_idx = np.where(fft_freq > 2000)[0]

        bass = np.mean(fft_data[bass_idx]) if len(bass_idx) > 0 else 0
        mid = np.mean(fft_data[mid_idx]) if len(mid_idx) > 0 else 0
        high = np.mean(fft_data[high_idx]) if len(high_idx) > 0 else 0

        bass = min(1.0, bass * 0.01)
        mid = min(1.0, mid * 0.01)
        high = min(1.0, high * 0.01)

        self.bass_history.append(bass)
        self.mid_history.append(mid)
        self.high_history.append(high)

        dominant_freq_idx = np.argmax(fft_data)
        freq_ratio = dominant_freq_idx / len(fft_data) if len(fft_data) > 0 else 0
        self.freq_history.append(freq_ratio)


    def start_audio_stream(self):
        self.stream = self.p.open(
            format = self.FORMAT,
            channels = self.CHANNELS,
            rate = self.RATE,
            input = True,
            frames_per_buffer = self.CHUNK,
            stream_callback = self.audio_callback
        )

        self.stream.start_stream()


    def draw_rounded_rect(self, surface, rect, color, radius = 10):
        rect = pygame.Rect(rect)

        pygame.draw.rect(surface, color, rect.inflate(-radius * 2, 0))
        pygame.draw.rect(surface, color, rect.inflate(0, -radius * 2))

        circles = [
            (rect.left + radius, rect.top + radius),
            (rect.right - radius, rect.top + radius),
            (rect.left + radius, rect.bottom - radius),
            (rect.right - radius, rect.bottom - radius)
        ]

        for circle_pos in circles:
            pygame.draw.circle(surface, color, circle_pos, radius)


    def draw_circle(self, center, radius, color, thickness = 1):
        if thickness <= 1:
            gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), int(radius), color)
        else:
            for i in range(thickness):
                gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), int(radius - i), color)


    def draw_visualization(self):
        current_ui_width = int(self.ui_width * self.ui_animation_progress)
        center_x = (self.screen_width + current_ui_width) // 2
        center_y = self.screen_height // 2

        volume = self.volume_history[-1] if self.volume_history else 0
        bass   = self.bass_history[-1] if self.bass_history else 0
        mid    = self.mid_history[-1] if self.mid_history else 0
        high   = self.high_history[-1] if self.high_history else 0

        base_radius = 100 + (volume * 300)
        thickness = 1 + int(bass * 20)

        for i in range(3):
            circle_radius = base_radius * (0.5 + i * 0.2) * (1.0 + high * 0.5)
            self.draw_circle((center_x, center_y), circle_radius, self.fg_color, thickness)

        if len(self.bass_history) > 2:
            for i, (b, m, h) in enumerate(zip(
                list(self.bass_history)[-50:], 
                list(self.mid_history)[-50:], 
                list(self.high_history)[-50:]
            )):
                angle = np.pi * i / 49
                distance = 80 + (b + m + h) * 50

                x = center_x + np.cos(angle) * distance * 3
                y = center_y + np.sin(angle) * distance

                size = 1 + (b + m + h) * 15

                pygame.draw.circle(self.screen, self.fg_color, (int(x), int(y)), int(size))


    def draw_toggle_hint(self):
        if self.ui_animation_progress < 0.1:
            hint_text = 'Arayüzü görmek için "H" tuşuna basın.'
            hint_surface = self.font.render(hint_text, True, (150, 150, 150))
            hint_rect = hint_surface.get_rect(topright = (self.screen_width - 10, 10))

            bg_rect = hint_rect.inflate(20, 10)

            s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            s.fill((20, 20, 20, 150))

            self.screen.blit(s, bg_rect)
            self.screen.blit(hint_surface, hint_rect)


    def draw_modern_ui(self):
        if self.ui_animation_active:
            if self.ui_animation_target > self.ui_animation_progress:
                self.ui_animation_progress = min(self.ui_animation_target, self.ui_animation_progress + self.ui_animation_speed)
            else:
                self.ui_animation_progress = max(self.ui_animation_target, self.ui_animation_progress - self.ui_animation_speed)

            if abs(self.ui_animation_progress - self.ui_animation_target) < 0.01:
                self.ui_animation_progress = self.ui_animation_target
                self.ui_animation_active = False

        if self.ui_animation_progress <= 0:
            self.draw_toggle_hint()
            return

        current_ui_width = int(self.ui_width * self.ui_animation_progress)

        ui_panel = pygame.Surface((current_ui_width, self.screen_height), pygame.SRCALPHA)
        ui_panel.fill((20, 20, 20, 200))

        if current_ui_width < 50:
            self.screen.blit(ui_panel, (0, 0))
            return

        content_opacity = int(min(255, 255 * (self.ui_animation_progress * 1.2)))

        title_text = self.title_font.render("SES PENCERESİ", True, (*self.fg_color[:3], content_opacity))
        ui_panel.blit(title_text, (20, 20))

        toggle_text = self.font.render('Arayüzü gizlemek için "H" tuşuna basın.', True, (150, 150, 150, content_opacity))
        ui_panel.blit(toggle_text, (20, 50))

        pygame.draw.line(ui_panel, (100, 100, 100, content_opacity), (20, 75), (current_ui_width - 20, 75), 1)

        fps_text = f""
        fps_surface = self.font.render(fps_text, True, (*self.fg_color[:3], content_opacity))
        ui_panel.blit(fps_surface, (20, 85))

        volume = self.volume_history[-1] if self.volume_history else 0
        bass   = self.bass_history[-1] if self.bass_history else 0
        mid    = self.mid_history[-1] if self.mid_history else 0
        high   = self.high_history[-1] if self.high_history else 0

        y_pos = 120
        metric_height = 70

        for label, value in [
            ("SES SEVİYESİ", volume),
            ("BAS SEVİYESİ", bass),
            ("YÜKSEKLİK ARALIĞI", high)
        ]:
            label_surface = self.font.render(label, True, (*self.fg_color[:3], content_opacity))

            ui_panel.blit(label_surface, (20, y_pos))

            value_text    = f"{value:.2f}"
            value_surface = self.font.render(value_text, True, (*self.fg_color[:3], content_opacity))
            value_pos_x   = min(current_ui_width - 20 - value_surface.get_width(), 250 - value_surface.get_width())

            ui_panel.blit(value_surface, (value_pos_x, y_pos))

            bar_width = min(260, current_ui_width - 40)
            bar_rect  = pygame.Rect(20, y_pos + 25, bar_width, 8)

            pygame.draw.rect(ui_panel, (50, 50, 50, content_opacity), bar_rect, border_radius = 4)

            fill_width = int(bar_width * value)

            if fill_width > 0:
                fill_rect = pygame.Rect(20, y_pos + 25, fill_width, 8)
                pygame.draw.rect(ui_panel, (*self.fg_color[:3], content_opacity), fill_rect, border_radius = 4)

            y_pos += metric_height

        if current_ui_width >= 200:
            info_y = y_pos + 20
            info_text = [
                "ÇALIŞMA PRENSİBİ",
                "",
                "- Daire boyutu: Ses Seviyesi",
                "- Daire kalınlığı: Bas seviyesi",
                "- Daire genişlemesi: Yükseklik aralığı",
                "- Parçacıklar: Frekans dağılımı"
            ]

            info_height = len(info_text) * 20 + 40
            info_rect = pygame.Rect(10, info_y - 10, current_ui_width - 20, info_height)

            self.draw_rounded_rect(ui_panel, info_rect, (40, 40, 40, content_opacity))

            for i, text in enumerate(info_text):
                if i == 0:
                    text_surface = self.title_font.render(text, True, (*self.fg_color[:3], content_opacity))
                else:
                    text_surface = self.font.render(text, True, (*self.fg_color[:3], content_opacity))

                ui_panel.blit(text_surface, (20, info_y + i * 20))

            controls_y = info_y + info_height + 20
            controls_text = [
                "KLAVYE KONTROLLERİ",
                "",
                '"H" Arayüzü gizler.',
                '"ESC" Uygulamadan çıkar.'
            ]

            controls_height = len(controls_text) * 20 + 40
            controls_rect   = pygame.Rect(10, controls_y - 10, current_ui_width - 20, controls_height)
            self.draw_rounded_rect(ui_panel, controls_rect, (40, 40, 40, content_opacity))

            for i, text in enumerate(controls_text):
                if i == 0:
                    text_surface = self.title_font.render(text, True, (*self.fg_color[:3], content_opacity))
                else:
                    text_surface = self.font.render(text, True, (*self.fg_color[:3], content_opacity))

                ui_panel.blit(text_surface, (20, controls_y + i * 20))

        if current_ui_width >= 150:
            attribution_y = self.screen_height - 40
            attr_text = "Erdemli Borsa İstanbul Fen Lisesi"
            attr_surface = self.font.render(attr_text, True, (150, 150, 150, content_opacity))

            ui_panel.blit(attr_surface, (20, attribution_y))

        self.screen.blit(ui_panel, (0, 0))

        if self.ui_animation_progress < 0.5:
            self.draw_toggle_hint()


    def toggle_ui(self):
        if self.toggle_cooldown <= 0:
            self.show_ui = not self.show_ui
            self.ui_animation_target = 1.0 if self.show_ui else 0.0
            self.ui_animation_active = True
            self.toggle_cooldown = 10


    def update_animation(self):
        if self.toggle_cooldown > 0:
            self.toggle_cooldown -= 1


    def run(self):
        self.running = True
        self.start_audio_stream()

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_h:
                            self.toggle_ui()

                self.update_animation()

                self.screen.fill(self.bg_color)

                self.draw_visualization()
                self.draw_modern_ui()

                pygame.display.flip()

                self.clock.tick(60)

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            pygame.quit()


if __name__ == "__main__":
    visualizer = AudioVisualizer()
    visualizer.run()
