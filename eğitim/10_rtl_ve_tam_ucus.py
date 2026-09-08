import asyncio
from mavsdk import System

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"
UCUS_IRTIFASI = 50.0

async def baglan_ve_hazirla(drone):
    await drone.connect(system_address=BAGLANTI_ADRESI)
    print("Baglanti bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-> Baglandi.")
            break

    print("Konum tahmini bekleniyor...")
    async for saglik in drone.telemetry.health():
        if saglik.is_global_position_ok and saglik.is_home_position_ok:
            print("-> Sistem ucusa hazir.\n")
            break

async def kalkis(drone):
    print("ARM ediliyor...")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(UCUS_IRTIFASI)
    print("Kalkis...")
    await drone.action.takeoff()
    async for konum in drone.telemetry.position():
        if konum.relative_altitude_m >= UCUS_IRTIFASI * 0.95:
            print("-> Ucus irtifasinda.\n")
            break
        await asyncio.sleep(1)

async def bir_noktaya_git(drone):
    async for konum in drone.telemetry.position():
        lat0 = konum.latitude_deg
        lon0 = konum.longitude_deg
        zemin_amsl = konum.absolute_altitude_m - konum.relative_altitude_m
        break

    hedef_lat = lat0 + 0.003
    hedef_amsl = zemin_amsl + UCUS_IRTIFASI
    print("Bir noktaya gidiliyor...")
    await drone.action.goto_location(hedef_lat, lon0, hedef_amsl, float("nan"))
    await asyncio.sleep(30)
    print("-> Nokta hedeflendi.\n")

async def eve_don_ve_in(drone):
    print("RTL: Kalkis noktasina donuluyor...")
    await drone.action.return_to_launch()

    print("Inis izleniyor...")
    async for havada in drone.telemetry.in_air():
        if not havada:
            print("-> Arac guvenle indi.")
            break

    async for armli in drone.telemetry.armed():
        print(f"-> Arm durumu: {armli}")
        break

async def main():
    drone = System()

    await baglan_ve_hazirla(drone)
    await kalkis(drone)
    await bir_noktaya_git(drone)
    await eve_don_ve_in(drone)

    print("\n============================================")
    print(" TEBRIKLER! Tam ucus senaryosu tamamlandi.")
    print(" Tum MAVSDK temel derslerini bitirdiniz.")
    print("============================================")

if __name__ == "__main__":
    asyncio.run(main())
