import asyncio
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan

BAGLANTI_ADRESI = "udpin://0.0.0.0:14540"
GOREV_IRTIFASI = 50.0
GOREV_HIZI = 15.0

def waypoint(lat, lon):
    return MissionItem(
        latitude_deg=lat,
        longitude_deg=lon,
        relative_altitude_m=GOREV_IRTIFASI,
        speed_m_s=GOREV_HIZI,
        is_fly_through=True,
        gimbal_pitch_deg=float("nan"),
        gimbal_yaw_deg=float("nan"),
        camera_action=MissionItem.CameraAction.NONE,
        loiter_time_s=float("nan"),
        camera_photo_interval_s=float("nan"),
        acceptance_radius_m=50.0,
        yaw_deg=float("nan"),
        camera_photo_distance_m=float("nan"),
        vehicle_action=MissionItem.VehicleAction.NONE,
    )

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
        lat0 = konum.latitude_deg
        lon0 = konum.longitude_deg
        break

    d = 0.003
    noktalar = [
        waypoint(lat0 + d, lon0),
        waypoint(lat0 + d, lon0 + d),
        waypoint(lat0,     lon0 + d),
        waypoint(lat0,     lon0),
    ]
    plan = MissionPlan(noktalar)

    await drone.mission.set_return_to_launch_after_mission(True)

    print(f"\n{len(noktalar)} waypoint yukleniyor (~300 m dortgen)...")
    await drone.mission.upload_mission(plan)
    print("-> Gorev yuklendi.")

    print("ARM ediliyor...")
    await drone.action.arm()
    print("Gorev baslatiliyor...")
    await drone.mission.start_mission()

    print("Gorev ilerlemesi:")
    async for ilerleme in drone.mission.mission_progress():
        print(f"  {ilerleme.current}/{ilerleme.total}")
        if ilerleme.current == ilerleme.total:
            print("-> Tum waypoint'ler tamamlandi.")
            break

    print("\nSabit kanat gorev/mission dersi tamamlandi (RTL otomatik).")

if __name__ == "__main__":
    asyncio.run(main())
