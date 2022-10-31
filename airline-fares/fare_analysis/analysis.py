import csv
from datetime import datetime

import pandas

from airport import Airport


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

    dateDiff = []
    for search, flight in zip(df['searchDate'], df['flightDate']):
        dt = flight - search
        dateDiff.append(dt.days)
    df['dateDiff'] = dateDiff


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


def correlation_price_date(df, travel_class, airlines=None):

    def filter_by_class(series):
        def filter_one(record):
            return all(map(lambda x: x == travel_class, record.split('||')))
        return series.apply(filter_one)

    filtered = df.copy()
    filtered = df[filter_by_class(df['segmentsCabinCode'])]
    filtered = df[df['dateDiff'].notna()]
    filtered = df[df['totalFare'].notna()]

    filtered['dateDiff'] = 365 - filtered['dateDiff']

    if airlines:
        def filter_by_airline(series):
            def filter_one(al):
                return all(map(lambda x: x in airlines, al.split('||')))
            return series.apply(filter_one)
        filtered = filtered[filter_by_airline(filtered['segmentsAirlineCode'])]

    print(filtered.head())

    corr = filtered['totalFare'].corr(filtered['dateDiff'])

    res_str = f'Correlation for travel class {travel_class}'
    if airlines:
        res_str += f' and airlines {airlines}'
    res_str += f': {corr}'

    print(res_str)


def scraped_more_than_day_before(df):
    count = sum(map(lambda x: x > 1, df['dateDiff']))
    print(f'#deals scraped more than one day before day of flight: {count}')


def process():
    print('Loading dataset...')
    df = load_df('data/sample.csv')

    print('Loading airport data...')
    airports = load_airports('data/airport-codes_csv.csv')

    print('Preprocessing dataset...')
    airlines = extract_airlines(df)

    preprocess_df(df)

    airlines_flying_together(df)
    ap = most_frequented_airports(df, 10, airports)[:5]
    most_desired_destination(df, 10, airports)[:5]
    most_desired_route(df, 10, airports)[:5]

    correlation_price_date(df, 'coach')
    correlation_price_date(df, 'business')
    correlation_price_date(df, 'coach', ['DL', 'AA', 'UA'])
    correlation_price_date(df, 'business', ['DL', 'AA', 'UA'])
    correlation_price_date(df, 'coach', airlines.keys() - ['DL', 'AA', 'UA'])
    correlation_price_date(df, 'business',
                           airlines.keys() - ['DL', 'AA', 'UA'])
    scraped_more_than_day_before(df)
