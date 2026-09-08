import asyncio
from mavsdk import System

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"
KALKIS_IRTIFASI = 50.0

async def main():
    drone = System()
    await drone.connect(system_address=BAGLANTI_ADRESI)

    print("Baglanti bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-> Baglandi.")
            break

    print("Konum tahmini bekleniyor...")
    async for saglik in drone.telemetry.health():
        if saglik.is_global_position_ok and saglik.is_home_position_ok:
            print("-> Hazir.")
            break

    print("\nARM ediliyor...")
    await drone.action.arm()
    print("-> ARM tamam.")

    await drone.action.set_takeoff_altitude(KALKIS_IRTIFASI)
    print(f"Kalkis irtifasi ayarlandi: {KALKIS_IRTIFASI} m")

    print("Kalkis komutu veriliyor (ucak tirmanacak)...")
    await drone.action.takeoff()

    print("Tirmaniyor...")
    async for konum in drone.telemetry.position():
        irtifa = konum.relative_altitude_m
        print(f"  Irtifa: {irtifa:.1f} m")
        if irtifa >= KALKIS_IRTIFASI * 0.95:
            print("-> Hedef irtifaya ulasildi. Ucak otomatik daire cizecek.")
            break
        await asyncio.sleep(1)

    print("\n15 saniye LOITER (daire) yapiyor...")
    await asyncio.sleep(15)

    print("Kalkis noktasina donuluyor (RTL)...")
    await drone.action.return_to_launch()

    print("\nSabit kanat kalkis dersi tamamlandi.")

if __name__ == "__main__":
    asyncio.run(main())
