import asyncio
from mavsdk import System

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"

async def main():
    drone = System()

    print(f"Baglaniliyor -> {BAGLANTI_ADRESI}")
    await drone.connect(system_address=BAGLANTI_ADRESI)

    print("Aracin baglanmasi bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-> BAGLANTI KURULDU!")
            break

    print("Araç bilgisi okunuyor...")
    async for info in drone.info.get_identification():
        print(f"-> Arac UID: {info.hardware_uid}")
        break

    print("\nBaglanti dersi tamamlandi. Kanali kapatabilirsiniz (Ctrl+C).")

if __name__ == "__main__":
    asyncio.run(main())
