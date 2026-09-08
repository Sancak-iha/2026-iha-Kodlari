# MAVSDK Egitim Serisi — SABIT KANAT (Fixed-Wing)

Her dosya **tek bir konuyu** ogretir. Sirayla ilerleyin.
Tum dersler **sabit kanat** ucak icin hazirlanmistir.

## Sabit kanat = multikopter DEGIL
Bu serideki en onemli farklar:
- **Havada asili kalinamaz.** HOLD/loiter = bir nokta etrafinda **daire** cizmek.
- **0 hiz = STALL (dusme).** Ucak surekli ileri hava hizina ihtiyac duyar.
- **Dikey kalkis/inis yok.** Kalkis egimle tirmanis; inis suzulerek pist yaklasimi.
- **Genis donusler.** Waypoint'ler ve hedef noktalar **yuzlerce metre** aralikli olmali.
- **Yuksek irtifa/hiz.** Ornek: irtifa 50 m, cruise 15 m/s.

## Kurulum
```bash
pip install mavsdk
```

## Simulasyon (once bunu baslatin)
PX4 SITL **sabit kanat** modeliyle calismali
(varsayilan baglanti: `udpin://0.0.0.0:14540`).
```bash
make px4_sitl gz_rc_cessna     # sabit kanat (Gazebo)
# veya:  make px4_sitl gazebo-classic_plane
```

## Dersler

| # | Dosya | Konu |
|---|-------|------|
| 01 | `01_baglanti.py` | Araca baglanma (connection) |
| 02 | `02_telemetri.py` | Canli veri okuma: konum, batarya, saglik |
| 03 | `03_arm.py` | Motorlari arm / disarm etme |
| 04 | `04_mod_degistirme.py` | Ucus modu degistirme (HOLD = loiter/daire) |
| 05 | `05_kalkis.py` | Otomatik kalkis (egimle tirmanis) |
| 06 | `06_inis.py` | Suzulerek inis ve inis dogrulama (land) |
| 07 | `07_git_konuma.py` | Uzak bir GPS noktasina gitme + loiter (goto) |
| 08 | `08_gorev_mission.py` | Genis waypoint gorevi (mission) |
| 09 | `09_offboard.py` | Bilgisayardan **konum** setpoint kontrolu (offboard) |
| 10 | `10_rtl_ve_tam_ucus.py` | RTL + tum dersleri birlestiren tam ucus |

## Calistirma
```bash
python 01_baglanti.py
```

## UYARILAR
- Once **her zaman simulasyonda** deneyin.
- Gercek ucakta pervane takiliyken arm/kalkis komutlarini test etmeyin
  (arm = motor doner; el firlatmali sistemlerde tehlikeli).
- **Sabit kanata asla "0 hiz / dur" komutu vermeyin** -> stall/dusme.
  Kontrolu konum (position) setpoint'i ile yapin (bkz. Ders 09).
- `goto_location` irtifasi **AMSL** (deniz seviyesi), `takeoff` irtifasi
  kalkis noktasina goredir. Karistirmayin (bkz. Ders 07).
- Offboard `start()` oncesi mutlaka bir setpoint gonderin (bkz. Ders 09).
- Waypoint/hedef noktalari **yeterince uzak** secin; ucak dar donemez.
