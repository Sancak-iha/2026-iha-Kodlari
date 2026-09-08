import asyncio
from mavsdk import System

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"
UCUS_IRTIFASI = 50.0
KUZEYE_KAYMA = 0.003

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

    async for konum in drone.telemetry.position():
        baslangic_lat = konum.latitude_deg
        baslangic_lon = konum.longitude_deg
        zemin_amsl = konum.absolute_altitude_m - konum.relative_altitude_m
        break
    print(f"Baslangic: {baslangic_lat:.7f}, {baslangic_lon:.7f}")

    print("\nKalkis...")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(UCUS_IRTIFASI)
    await drone.action.takeoff()
    async for konum in drone.telemetry.position():
        if konum.relative_altitude_m >= UCUS_IRTIFASI * 0.95:
            break
        await asyncio.sleep(1)
    print("-> Ucus irtifasinda.\n")

    hedef_lat = baslangic_lat + KUZEYE_KAYMA
    hedef_lon = baslangic_lon
    hedef_amsl = zemin_amsl + UCUS_IRTIFASI

    print(f"Hedefe gidiliyor: {hedef_lat:.7f}, {hedef_lon:.7f}")
    await drone.action.goto_location(
        hedef_lat, hedef_lon, hedef_amsl, float("nan")
    )

    print("Yol izleniyor (sabit kanat yavas ilerler)...")
    for _ in range(45):
        async for konum in drone.telemetry.position():
            kalan_lat = abs(konum.latitude_deg - hedef_lat)
            print(f"  Kalan enlem farki: {kalan_lat:.6f}")
            break
        if kalan_lat < 0.0005:
            print("-> Hedef bolgeye ulasildi (ucak daire cizecek).")
            break
        await asyncio.sleep(1)

    print("\nKalkis noktasina donuluyor (RTL)...")
    await drone.action.return_to_launch()

    print("\nGoto dersi tamamlandi.")

if __name__ == "__main__":
    asyncio.run(main())
