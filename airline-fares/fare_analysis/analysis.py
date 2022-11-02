import csv
from datetime import datetime

import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from airport import Airport
from flight import Flight
from aircraft import Aircraft


def parse_date(date: str):
    format = '%Y-%m-%d'
    return datetime.strptime(date, format)


def load_df(path: str):
    return pandas.read_csv(path, sep=',')


def load_airports(path: str) -> dict[str, Airport]:

    airports = dict()

    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            if not row[1] or row[1] in ('closed', 'heliport') or \
               len(row[0]) != 4 or not row[9]:
                continue

            icao = row[0]
            iata = row[9]
            name = row[2]
            country = row[5]

            airport = Airport(
                    icao=icao,
                    iata=iata,
                    name=name,
                    country=country,
                    )

            airports[iata] = airport
            airports[icao] = airport

    return airports


def airlines_flying_together(df) -> dict[str, set[str]]:
    together = dict()

    for flight in df['segmentsAirlineCode']:
        airlines = flight.split('||')

        for airline in airlines:
            shared_flights = together.get(airline, set())
            for a2 in airlines:
                if a2 == airline:
                    continue
                shared_flights.add(a2)

            together[airline] = shared_flights

    print('|airline|shares flights with|')
    print('|-------|-------------------|')
    for airline, other in together.items():
        print(f'|{airline}|{", ".join(other)}|')


def appearances_in_multisegment(df) -> dict[str, set[str]]:
    appearances = dict()

    for flight in df['segmentsAirlineCode']:
        airlines = flight.split('||')

        for airline in airlines:
            appearances[airline] = appearances.get(airline, 0) + 1

    print('|airline|appearances in multisegment trips|')
    print('|-------|---------------------------------|')
    for airline, count in appearances.items():
        print(f'|{airline}|{count}|')


def extract_airlines(df):
    airlines = dict()

    names = df['segmentsAirlineName']
    codes = df['segmentsAirlineCode']

    for name, iata in zip(names, codes):

        names_split = name.split('||')
        iata_split = iata.split('||')

        for name, iata in zip(names_split, iata_split):
            airlines[iata] = name

    return airlines


def preprocess_df(df):
    df.drop(columns=['legId',
                     'segmentsAirlineName',
                     'segmentsDepartureTimeRaw',
                     'segmentsArrivalTimeRaw',
                     ])
    df['searchDate'] = df['searchDate'].apply(parse_date)
    df['flightDate'] = df['flightDate'].apply(parse_date)
    df['segmentsEquipmentDescription'] = \
        df['segmentsEquipmentDescription'].astype(str)

    dateDiff = []
    for search, flight in zip(df['searchDate'], df['flightDate']):
        dt = flight - search
        dateDiff.append(dt.days)
    df['dateDiff'] = dateDiff

    assign_buckets(df)


def most_frequented_airports(df, limit, airport_data):
    destination = df['destinationAirport']
    departures = df['segmentsDepartureAirportCode']

    airports = dict()

    for dest, dep in zip(destination, departures):
        visited = dep.split('||')
        visited.append(dest)
        for v in visited:
            airports[v] = airports.get(v, 0) + 1

    most_freq = sorted(airports.items(),
                       key=lambda x: x[1],
                       reverse=True)[:limit]

    print('|Airport|Airport name|Number of visits|')
    print('|-------|------------|----------------|')
    for airport, visits in most_freq:
        print(f'|{airport}|{airport_data[airport].name}|{visits}|')

    return most_freq


def most_desired_destination(df, limit, airport_data):
    destination = df['destinationAirport']
    frequency = dict()

    for ap in destination:
        frequency[ap] = frequency.get(ap, 0) + 1

    most_freq = sorted(frequency.items(),
                       key=lambda x: x[1],
                       reverse=True)[:limit]

    print('|Airport|Airport name|Number of trips|')
    print('|-------|------------|----------------|')
    for airport, trips in most_freq:
        print(f'|{airport}|{airport_data[airport].name}|{trips}|')

    return most_freq


def most_desired_route(df, limit, airport_data):
    departure = df['segmentsDepartureAirportCode']
    destination = df['destinationAirport']

    frequency = dict()

    for key in zip(departure, destination):
        frequency[key] = frequency.get(key, 0) + 1

    most_freq = sorted(frequency.items(),
                       key=lambda x: x[1],
                       reverse=True)[:limit]

    print('|Route|Number of trips|')
    print('|-----|----------------|')
    for route, trips in most_freq:
        dep, dest = route
        print(f'|{dep}-{dest}|{trips}|')

    return most_freq


def filter_by_airline(series, airlines):
    def filter_one(al):
        return all(map(lambda x: x in airlines, al.split('||')))
    return series.apply(filter_one)


def filter_by_class(series, travel_class):
    def filter_one(record):
        return all(map(lambda x: x == travel_class, record.split('||')))
    return series.apply(filter_one)


def correlation_price_date(df, travel_class, airlines=None):

    filtered = df.copy()
    filtered = filtered[filter_by_class(filtered['segmentsCabinCode'],
                                        travel_class)]
    filtered = filtered[filtered['dateDiff'].notna()]
    filtered = filtered[filtered['totalFare'].notna()]

    filtered['dateDiff'] = 365 - filtered['dateDiff']

    if airlines:
        filtered = filtered[filter_by_airline(filtered['segmentsAirlineCode'],
                                              airlines)]

    corr = filtered['totalFare'].corr(filtered['dateDiff'])

    res_str = f'Correlation for travel class {travel_class}'
    if airlines:
        res_str += f' and airlines {airlines}'
    res_str += f': {corr}'

    print(res_str)


def scraped_more_than_day_before(df):
    count = sum(map(lambda x: x > 1, df['dateDiff']))
    print(f'#deals scraped more than one day before day of flight: {count}')


date_diff_buckets = [
            range(2), range(2, 4), range(4, 8), range(8, 15), range(15, 22),
            range(22, 31), range(31, 62), range(62, 367),
            ]


def assign_buckets(df):
    def assign_one(diff):
        for i, b in enumerate(date_diff_buckets):
            if diff in b:
                return i
        raise RuntimeError('Delta outside of bucket range')

    df['dateDiffBucket'] = df['dateDiff'].apply(assign_one)


def ticket_price_by_bucket(df, travel_class, airlines):

    filtered = df.copy()
    filtered = filtered[filter_by_class(filtered['segmentsCabinCode'],
                                        travel_class)]
    filtered = filtered[filtered['dateDiff'].notna()]
    filtered = filtered[filtered['totalFare'].notna()]

    if airlines:
        filtered = filtered[filter_by_airline(filtered['segmentsAirlineCode'],
                                              airlines)]

    plt.figure()
    sns.boxplot(data=filtered, y='totalFare', x='dateDiffBucket') \
        .set(title=f'{travel_class} ({", ".join(airlines)})')
    plt.savefig(f'plots/buckets_{travel_class}_{"_".join(airlines)}.png')


def ticket_price_by_dow(df, travel_class, airlines):
    filtered = df.copy()
    filtered = filtered[filter_by_class(filtered['segmentsCabinCode'],
                                        travel_class)]
    filtered = filtered[filtered['dateDiff'].notna()]
    filtered = filtered[filtered['totalFare'].notna()]

    if airlines:
        filtered = filtered[filter_by_airline(filtered['segmentsAirlineCode'],
                                              airlines)]

    filtered['dow'] = filtered['flightDate'].apply(lambda x: x.weekday())

    plt.figure()
    sns.boxplot(data=filtered, y='totalFare', x='dow') \
        .set(title=f'{travel_class} ({", ".join(airlines)})')
    plt.savefig(f'plots/dow_{travel_class}_{"_".join(airlines)}.png')


empty_seats_buckets = [
        range(2), range(2, 5), range(5, 10), range(10, 20), range(20, 1000),
        ]


def ticket_price_by_empty_seats(df, travel_class, airlines):
    filtered = df.copy()
    filtered = filtered[filter_by_class(filtered['segmentsCabinCode'],
                                        travel_class)]
    filtered = filtered[filtered['dateDiff'].notna()]
    filtered = filtered[filtered['totalFare'].notna()]

    if airlines:
        filtered = filtered[filter_by_airline(filtered['segmentsAirlineCode'],
                                              airlines)]

    def assign_one(seats):
        for i, b in enumerate(empty_seats_buckets):
            if seats in b:
                return i
        raise RuntimeError('Num of empty seats outside of bucket range')

    filtered['emptySeatsBucket'] = filtered['seatsRemaining'].apply(assign_one)

    plt.figure()
    sns.boxplot(data=filtered, y='totalFare', x='emptySeatsBucket') \
        .set(title=f'{travel_class} ({", ".join(airlines)})')
    file = f'plots/seatsRemaining_{travel_class}_{"_".join(airlines)}.png'
    plt.savefig(file)


def offers_by_date_diff(df, travel_class):
    filtered = df.copy()
    filtered = filtered[filter_by_class(filtered['segmentsCabinCode'],
                                        travel_class)]
    filtered = filtered[filtered['dateDiff'].notna()]
    filtered = filtered[filtered['totalFare'].notna()]

    def carriers_to_type(airlines):
        split = airlines.split('||')

        tc = {'AA', 'DL', 'UA'}
        lc = {'B6', 'NK', 'SY', 'F9'}
        trad = all(map(lambda x: x in tc, split))
        lcc = all(map(lambda x: x in lc, split))

        if trad:
            return 'traditional'
        elif lcc:
            return 'lcc'
        else:
            return 'ignore'

    filtered['carrierType'] = filtered['segmentsAirlineCode'] \
        .apply(carriers_to_type)

    filtered = filtered[filtered['carrierType'] != 'ignore']

    plt.figure()
    sns.boxplot(data=filtered, y='dateDiff', x='carrierType') \
        .set(title=f'{travel_class}')
    file = f'plots/offers_by_date_diff_{travel_class}.png'
    plt.savefig(file)


def parse_aircraft(equipment: str):
    equipment = equipment.lower()

    manufacturer = ''
    model = ''

    if not equipment:
        return None

    if 'airbus' in equipment:
        manufacturer = 'Airbus'
        if 'a220' in equipment:
            model = 'A220'
        elif 'a300' in equipment:
            model = 'A300'
        elif 'a310' in equipment:
            model = 'A310'
        elif 'a330' in equipment:
            model = 'A330'
        elif 'a340' in equipment:
            model = 'A340'
        elif 'a350' in equipment:
            model = 'A350'
        elif 'a380' in equipment:
            model = 'A380'
        elif 'a318' in equipment or 'a319' in equipment or 'a32' in equipment:
            model = 'A320'
        else:
            raise RuntimeError(f'Unexpected aircraft{equipment}')
    elif 'boeing' in equipment:
        manufacturer = 'Boeing'
        if '707' in equipment:
            model = '707'
        elif '717' in equipment:
            model = '717'
        elif '727' in equipment:
            model = '727'
        elif '737' in equipment:
            model = '737'
        elif '747' in equipment:
            model = '747'
        elif '757' in equipment:
            model = '757'
        elif '767' in equipment:
            model = '767'
        elif '777' in equipment:
            model = '777'
        elif '787' in equipment:
            model = '787'
        else:
            raise RuntimeError(f'Unexpected aircraft{equipment}')
    elif 'embraer' in equipment:
        manufacturer = 'Embraer'
        if '175' in equipment or '170' in equipment:
            model = '175'
        elif '190' in equipment or '195' in equipment:
            model = '190'
        elif '145' in equipment:
            model = '145'
        else:
            raise RuntimeError(f'Unexpected aircraft{equipment}')
    elif 'canadair' in equipment or 'canadian' in equipment:
        manufacturer = 'CRJ'
        model = '900'
        if '700' in equipment:
            model = '700'
    # Unfortunately, we do not know which Cessna we're flying, thus we can only
    # choose to ignore it
    elif 'metro' in equipment or 'bus' in equipment or 'varies' in equipment \
            or equipment == 'nan' or equipment == 'cessna':
        return None
    elif 'tecnam' in equipment:
        manufacturer = 'Tecnam'
        model = 'p2012'
    elif 'pilatus' in equipment:
        manufacturer = 'Pilatus'
        model = 'PC-12'
    elif 'dehavilland' in equipment:
        manufacturer = 'Dehavilland'
        model = 'DHC-8'
    elif 'fairchild' in equipment:
        manufacturer = 'Fairchild'
        model = 'Dornier 238'
    elif 'atr' in equipment:
        manufacturer = 'ATR'
        model = '42' if '42' in equipment else '72'
    else:
        raise RuntimeError(f'Unexpected aircraft {equipment}')

    return Aircraft(manufacturer, model)


def extract_flights(df):
    flights = set()

    for _, row in df.iterrows():

        times = row['segmentsDepartureTimeEpochSeconds'].split('||')
        airports = [row['startingAirport']] + \
            row['segmentsArrivalAirportCode'].split('||')
        airlines = row['segmentsAirlineCode'].split('||')
        equipment = row['segmentsEquipmentDescription'].split('||')

        for i in range(len(airports) - 1):
            f = Flight(
                    time=times[i],
                    origin=airports[i],
                    destination=airports[i+1],
                    airline=airlines[i],
                    aircraft=parse_aircraft(equipment[i]),
                    )
            if f.aircraft:
                flights.add(f)

    return flights


def aircraft_stats(flights: list[Flight]):

    manufacturers = dict()
    models = dict()

    for flight in flights:
        m = flight.aircraft.manufacturer
        model = str(flight.aircraft)
        manufacturers[m] = manufacturers.get(m, 0) + 1
        models[model] = models.get(model, 0) + 1

    print('|manufacturer|count|')
    print('|------------|-----|')
    for manufacturer, count in manufacturers.items():
        print(f'|{manufacturer}|{count}|')

    print('|model|count|')
    print('|------------|-----|')
    for model, count in models.items():
        print(f'|{model}|{count}|')


def price_per_mile(df, travel_class, airlines):
    filtered = df.copy()
    filtered = filtered[filtered['isNonStop']]
    filtered = filtered[filtered['totalTravelDistance'].notna()]
    filtered = filtered[filter_by_class(filtered['segmentsCabinCode'],
                                        travel_class)]
    filtered = filtered[filtered['totalFare'].notna()]

    if airlines:
        filtered = filtered[filter_by_airline(filtered['segmentsAirlineCode'],
                                              airlines)]

    filtered['ppm'] = filtered['totalFare'] / filtered['totalTravelDistance']

    corr = filtered['ppm'].corr(filtered['totalTravelDistance'])
    print('Correlation between travel distance and price per mile',
          f'for class {travel_class} and airlines {airlines}: {corr}')

    plt.figure()
    plt.scatter(y=filtered['ppm'], x=filtered['totalTravelDistance'])
    plt.title(f'{travel_class} ({", ".join(airlines)})')
    file = f'plots/price_per_mile_{travel_class}_{"_".join(airlines)}.png'
    plt.savefig(file)


def process():
    print('Loading dataset...')
    df = load_df('data/sample.csv')

    print('Loading airport data...')
    airports = load_airports('data/airport-codes_csv.csv')

    print('Preprocessing dataset...')
    # airlines = extract_airlines(df)

    preprocess_df(df)

    airlines_flying_together(df)
    appearances_in_multisegment(df)
    most_frequented_airports(df, 10, airports)
    most_desired_destination(df, 10, airports)
    most_desired_route(df, 10, airports)

    correlation_price_date(df, 'coach')
    correlation_price_date(df, 'business')
    correlation_price_date(df, 'coach', ['DL', 'AA', 'UA'])
    correlation_price_date(df, 'business', ['DL', 'AA', 'UA'])
    correlation_price_date(df, 'coach', ['B6', 'NK', 'SY', 'F9'])
    correlation_price_date(df, 'business', ['B6', 'NK', 'SY', 'F9'])
    scraped_more_than_day_before(df)

    ticket_price_by_bucket(df, 'coach', ['DL', 'AA', 'UA'])
    ticket_price_by_bucket(df, 'business', ['DL', 'AA', 'UA'])
    ticket_price_by_bucket(df, 'coach', ['B6', 'NK', 'SY', 'F9'])
    ticket_price_by_bucket(df, 'business', ['B6', 'NK', 'SY', 'F9'])

    ticket_price_by_dow(df, 'coach', ['DL', 'AA', 'UA'])
    ticket_price_by_dow(df, 'business', ['DL', 'AA', 'UA'])
    ticket_price_by_dow(df, 'coach', ['B6', 'NK', 'SY', 'F9'])
    ticket_price_by_dow(df, 'business', ['B6', 'NK', 'SY', 'F9'])

    ticket_price_by_empty_seats(df, 'coach', ['DL', 'AA', 'UA'])
    ticket_price_by_empty_seats(df, 'business', ['DL', 'AA', 'UA'])
    ticket_price_by_empty_seats(df, 'coach', ['B6', 'NK', 'SY', 'F9'])
    ticket_price_by_empty_seats(df, 'business', ['B6', 'NK', 'SY', 'F9'])

    offers_by_date_diff(df, 'coach')
    offers_by_date_diff(df, 'business')

    flights = extract_flights(df)
    aircraft_stats(flights)

    price_per_mile(df, 'coach', ['DL', 'AA', 'UA'])
    price_per_mile(df, 'business', ['DL', 'AA', 'UA'])
    price_per_mile(df, 'coach', ['B6', 'NK', 'SY', 'F9'])
    price_per_mile(df, 'business', ['B6', 'NK', 'SY', 'F9'])
