import asyncio
from mavsdk import System

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"
UCUS_IRTIFASI = 50.0

async def main():
    drone = System()
    await drone.connect(system_address=BAGLANTI_ADRESI)

    print("Baglanti bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-> Baglandi.")
            break

    async for saglik in drone.telemetry.health():
        if saglik.is_global_position_ok and saglik.is_home_position_ok:
            break

    print("\nHazirlik: kalkis yapiliyor...")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(UCUS_IRTIFASI)
    await drone.action.takeoff()

    async for konum in drone.telemetry.position():
        if konum.relative_altitude_m >= UCUS_IRTIFASI * 0.95:
            break
        await asyncio.sleep(1)
    print("-> Ucus irtifasinda. Simdi inis dersine geciyoruz.\n")

    print("LAND komutu veriliyor (ucak suzulerek alcalacak)...")
    await drone.action.land()

    print("Inis izleniyor...")
    async for havada in drone.telemetry.in_air():
        if not havada:
            print("-> Ucak yere indi (in_air = False).")
            break

    async for armli in drone.telemetry.armed():
        print(f"-> Arm durumu: {armli}")
        break

    print("\nSabit kanat inis dersi tamamlandi.")
    print("Kontrollu pist inisi icin -> DERS 08 (mission landing).")

if __name__ == "__main__":
    asyncio.run(main())
