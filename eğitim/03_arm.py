import asyncio
from mavsdk import System

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"

async def main():
    drone = System()
    await drone.connect(system_address=BAGLANTI_ADRESI)

    print("Baglanti bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-> Baglandi.")
            break

    print("Arac arm'a hazir hale gelene kadar bekleniyor...")
    async for saglik in drone.telemetry.health():
        if saglik.is_global_position_ok and saglik.is_home_position_ok:
            print("-> Konum tahmini hazir, arm edilebilir.")
            break

    print("\nARM ediliyor...")
    try:
        await drone.action.arm()
        print("-> ARM basarili. Motorlar devrede.")
    except Exception as hata:
        print(f"-> ARM basarisiz: {hata}")
        return

    print("3 saniye arm'li beklenecek...")
    await asyncio.sleep(3)

    print("\nDISARM ediliyor...")
    try:
        await drone.action.disarm()
        print("-> DISARM basarili. Motorlar durdu.")
    except Exception as hata:
        print(f"-> DISARM basarisiz: {hata}")

    print("\nArm/Disarm dersi tamamlandi.")

if __name__ == "__main__":
    asyncio.run(main())
