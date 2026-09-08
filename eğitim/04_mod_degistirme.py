import asyncio
from mavsdk import System

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"

async def mevcut_modu_yaz(drone, etiket):
    async for mod in drone.telemetry.flight_mode():
        print(f"  [{etiket}] Aktif mod: {mod}")
        break

async def main():
    drone = System()
    await drone.connect(system_address=BAGLANTI_ADRESI)

    print("Baglanti bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-> Baglandi.\n")
            break

    await mevcut_modu_yaz(drone, "BASLANGIC")

    print("\nHOLD moduna geciliyor...")
    try:
        await drone.action.hold()
        print("-> HOLD komutu gonderildi.")
    except Exception as hata:
        print(f"-> HOLD basarisiz: {hata}")

    await asyncio.sleep(2)
    await mevcut_modu_yaz(drone, "SONRA")

    print("\nMod degistirme dersi tamamlandi.")
    print("Ileri: TAKEOFF/MISSION/OFFBOARD modlari ilgili derslerde.")

if __name__ == "__main__":
    asyncio.run(main())
