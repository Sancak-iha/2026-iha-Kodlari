import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, OffboardError

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

    print("\nKalkis...")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(UCUS_IRTIFASI)
    await drone.action.takeoff()
    async for konum in drone.telemetry.position():
        if konum.relative_altitude_m >= UCUS_IRTIFASI * 0.95:
            break
        await asyncio.sleep(1)
    print("-> Ucus irtifasinda.\n")

    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, -UCUS_IRTIFASI, 0.0)
    )

    print("Offboard baslatiliyor...")
    try:
        await drone.offboard.start()
    except OffboardError as hata:
        print(f"-> Offboard baslatilamadi: {hata._result.result}")
        await drone.action.return_to_launch()
        return
    print("-> Offboard aktif.\n")

    print("Kuzeyde 300 m noktaya gidiliyor...")
    await drone.offboard.set_position_ned(
        PositionNedYaw(300.0, 0.0, -UCUS_IRTIFASI, 0.0)
    )
    await asyncio.sleep(25)

    print("Doguda 300 m noktaya gidiliyor...")
    await drone.offboard.set_position_ned(
        PositionNedYaw(300.0, 300.0, -UCUS_IRTIFASI, 90.0)
    )
    await asyncio.sleep(25)

    print("Offboard durduruluyor...")
    await drone.offboard.stop()
    print("Kalkis noktasina donuluyor (RTL)...")
    await drone.action.return_to_launch()

    print("\nSabit kanat offboard dersi tamamlandi.")

if __name__ == "__main__":
    asyncio.run(main())
