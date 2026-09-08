import asyncio
from mavsdk import System

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"

async def baglan():
    drone = System()
    await drone.connect(system_address=BAGLANTI_ADRESI)
    print("Baglanti bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-> Baglandi.\n")
            break
    return drone

async def main():
    drone = await baglan()

    print("KONUM:")
    async for konum in drone.telemetry.position():
        print(f"  Enlem     : {konum.latitude_deg:.7f}")
        print(f"  Boylam    : {konum.longitude_deg:.7f}")
        print(f"  Irtifa(rel): {konum.relative_altitude_m:.2f} m")
        break

    print("\nBATARYA:")
    async for bat in drone.telemetry.battery():
        print(f"  Voltaj    : {bat.voltage_v:.2f} V")
        print(f"  Kalan     : {bat.remaining_percent * 100:.0f} %")
        break

    print("\nSAGLIK:")
    async for saglik in drone.telemetry.health():
        print(f"  Gyro kalibre     : {saglik.is_gyrometer_calibration_ok}")
        print(f"  Akselerometre    : {saglik.is_accelerometer_calibration_ok}")
        print(f"  Pusula (mag)     : {saglik.is_magnetometer_calibration_ok}")
        print(f"  Yerel konum OK   : {saglik.is_local_position_ok}")
        print(f"  Global konum OK  : {saglik.is_global_position_ok}")
        break

    print("\nDURUM:")
    async for mod in drone.telemetry.flight_mode():
        print(f"  Ucus modu : {mod}")
        break
    async for armli in drone.telemetry.armed():
        print(f"  Arm mi?   : {armli}")
        break

    print("\nTelemetri dersi tamamlandi.")

if __name__ == "__main__":
    asyncio.run(main())
