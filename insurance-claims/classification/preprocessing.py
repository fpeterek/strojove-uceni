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
    pass


def parse_brake_type(brakes: str) -> int:
    1 if brakes == 'Disc' else 0


def parse_transmission_type(transmission: str) -> int:
    1 if transmission == 'Automatic' else 0


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

    ds['max_torque'].map(parse_torque)
    ds['max_power'].map(parse_power)

    ds['is_esc'].map(parse_bool)
    ds['is_adjustable_steering'].map(parse_bool)
    ds['is_tpms'].map(parse_bool)
    ds['is_parking_sensors'].map(parse_bool)
    ds['is_parking_camera'].map(parse_bool)
    ds['is_front_fog_lights'].map(parse_bool)
    ds['is_rear_window_wiper'].map(parse_bool)
    ds['is_rear_window_defogger'].map(parse_bool)
    ds['is_brake_assist'].map(parse_bool)
    ds['is_power_door_locks'].map(parse_bool)
    ds['is_central_locking'].map(parse_bool)
    ds['is_driver_seat_height_adjustable'].map(parse_bool)
    ds['is_day_night_rear_view_mirror'].map(parse_bool)
    ds['is_speed_alert'].map(parse_bool)

    ds['rear_brakes_type'].map(parse_brake_type)
    ds['transmission_type'].map(parse_transmission_type)

    return ds
