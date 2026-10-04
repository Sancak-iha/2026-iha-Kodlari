# Sancak İHA Yer İstasyonu

## Jetson tarafı
`yolo_otonom_gercek.py` ve `yer_istasyonu.py` aynı klasörde olmalı.

    pip3 install websockets
    python3 yolo_otonom_gercek.py --camera-dev 0 --no-gui --no-mission-edit --cam-pitch-deg 20

Uçuş kodu olmadan sadece yayını denemek için:

    python3 yer_istasyonu.py --camera-dev 0      # veya --test-desen

Açılan portlar: 8080 (kamera, MJPEG) ve 8765 (telemetri, WebSocket).
Jetson'da güvenlik duvarı varsa: `sudo ufw allow 8080/tcp && sudo ufw allow 8765/tcp`

## Laptop tarafı (masaüstü uygulaması)
Node.js 18+ gerekli.

    npm install        # internet varken BİR KEZ
    npm start          # uygulamayı aç

Kurulum dosyası üretmek için:

    npm run dist:win     # Windows: dist/ altında kurulum + portable exe
    npm run dist:linux   # Linux: AppImage + deb

Uygulamada Jetson IP'sini yazıp **Bağlan**'a bas. IP ve port hatırlanır.
Kaydedilen kareler: Resimler/Sancak Yer Istasyonu (hedef kilitlenince otomatik).