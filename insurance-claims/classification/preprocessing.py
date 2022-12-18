import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def one_hot_encode(ds, attr) -> pd.DataFrame:
    enc_seg = LabelBinarizer()
    enc_seg.fit(sorted(ds[attr].unique()))
    transformed = enc_seg.transform(ds[attr])
    transformed = pd.DataFrame(transformed)
    transformed.columns = \
        [f'{attr}_{val}' for val in sorted(ds[attr].unique())]
    ds = pd.concat([ds, transformed], axis=1).drop([attr], axis=1)

    return ds


def parse_torque(torque: str) -> float:
    torque = torque[:torque.find('Nm@')]
    return float(torque)


def parse_power(power: str) -> float:
    power = power[:power.find('bhp@')]
    return float(power)


def parse_bool(boolean: str) -> bool:
    return int(boolean == 'Yes')


def parse_brake_type(brakes: str) -> int:
    return int(brakes == 'Disc')


def parse_transmission_type(transmission: str) -> int:
    return int(transmission == 'Automatic')


def convert_with(ds, col, fn) -> pd.DataFrame:
    ds[col] = ds[col].map(fn)


def convert_to_bool(ds, col) -> pd.DataFrame:
    convert_with(ds, col, parse_bool)


def preprocess(ds) -> pd.DataFrame:
    cols = list(ds.columns)
    cols.remove('policy_id')
    cols.remove('model')
    cols.remove('engine_type')
    cols.remove('is_power_steering')

    ds = ds[cols]

    ds = one_hot_encode(ds, 'segment')
    ds = one_hot_encode(ds, 'area_cluster')
    ds = one_hot_encode(ds, 'make')
    ds = one_hot_encode(ds, 'fuel_type')
    ds = one_hot_encode(ds, 'steering_type')

    convert_with(ds, 'max_torque', parse_torque)
    convert_with(ds, 'max_power', parse_power)

    convert_to_bool(ds, 'is_esc')
    convert_to_bool(ds, 'is_adjustable_steering')
    convert_to_bool(ds, 'is_tpms')
    convert_to_bool(ds, 'is_parking_sensors')
    convert_to_bool(ds, 'is_parking_camera')
    convert_to_bool(ds, 'is_front_fog_lights')
    convert_to_bool(ds, 'is_rear_window_wiper')
    convert_to_bool(ds, 'is_rear_window_washer')
    convert_to_bool(ds, 'is_rear_window_defogger')
    convert_to_bool(ds, 'is_brake_assist')
    convert_to_bool(ds, 'is_power_door_locks')
    convert_to_bool(ds, 'is_central_locking')
    convert_to_bool(ds, 'is_driver_seat_height_adjustable')
    convert_to_bool(ds, 'is_day_night_rear_view_mirror')
    convert_to_bool(ds, 'is_ecw')
    convert_to_bool(ds, 'is_speed_alert')

    convert_with(ds, 'rear_brakes_type', parse_brake_type)
    convert_with(ds, 'transmission_type', parse_transmission_type)

    ds = ds.copy()

    return ds
