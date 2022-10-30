Data source: https://www.kaggle.com/datasets/dilwong/flightprices

Airports: https://datahub.io/core/airport-codes

## Do the offers include multisegment flights with airlines without a codeshare agreement?

We aggregate the data by looking at which airlines appeared alongside one another in
multisegment flights. By looking at the table entry for United Airlines, we can clearly
see UA appears in trips consisting of multiple legs where at least on leg is operated by
either American Airlines or Delta Airlines. Not only can we deduce that, as these three
major US carriers are main competitors to one another, these airlines likely do not have
a codeshare agreement, we can also check out their websites and see that our deduction
was correct.

This discovery speaks volumes about Expedia. We can clearly see that Expedia sells trips
consisting of flights with airlines without a codeshare agreement. A codeshare agreement
is important for the passenger -- if two airlines have a codeshare agreement, the passenger's
luggage will be handled by the airport personnel during a layover. Otherwise, the passenger
will have to pickup their luggage during the layover and check-in again at their next carrier's
counter. This is not only a great inconvenience, but also puts the passenger at risk of not
catching their connecting flight, if they get stuck in airport queues. Thus, we have learned
that passengers need to be wary of what trips Expedia tries to sell to them and ensure they
always have enough time to catch their connecting flights.

|airline|shares flights with|
|-------|-------------------|
|AA|9X, 9K, UA, B6, 4B, LF, KG|
|DL|AS, 9K, UA|
|UA|AA, 9X, 9K, HA, AS, 4B, DL, KG|
|9K|AA, UA, B6, AS, DL|
|AS|9X, 9K, UA, HA, DL|
|B6|AA, 9K, 3M|
|NK||
|9X|AA, AS, UA|
|SY||
|F9||
|KG|AA, UA|
|LF|AA|
|4B|AA, UA|
|HA|AS, UA|
|3M|B6|

## Most frequented airports

Rather surprisingly, the most visited airport in trips sold by Expedia is LAX. This statistic
does not reflect the entire air travel situation in the US. In reality, ATL is the most is the
most frequented airport in the US,
[surpassing LAX by almost 30 milion pax/year](https://www.aerotime.aero/articles/31886-top-10-biggest-airports-in-the-world-2021).

ORD, DEN and DFW are in reality also bigger than LAX.

The obvious explanation would then be that Expedia is only focused on leisurely travel
and Los Angeles is a big tourist destination.

It must be noted that the data is a collection of offers scraped from Expedia, not a collection
of trips that were actually sold. Thus, we must take this data with a grain of salt, although we
can at the very least assume the airlines are not operating empty flights. A lot of resources
is spent on optimizing routes served by airlines, and therefore we can assume the airports with
the most offers available are also the most desired destinations among holiday-goers.

|Airport|Airport name|Number of visits|
|-------|------------|----------------|
|LAX|Los Angeles International Airport|898365|
|ORD|Chicago O'Hare International Airport|859588|
|ATL|Hartsfield Jackson Atlanta International Airport|809379|
|CLT|Charlotte Douglas International Airport|752622|
|BOS|General Edward Lawrence Logan International Airport|716426|
|DFW|Dallas Fort Worth International Airport|707309|
|LGA|La Guardia Airport|680578|
|SFO|San Francisco International Airport|606993|
|JFK|John F Kennedy International Airport|588321|
|DEN|Denver International Airport|572937|

## Most frequent destination

If we look at the number of the most frequently offered destinations, we confirm our suspicions
from the previous section. Expedia offers a lot of trips where Los Angeles is the final
destination, meaning it is a destination favored by US citizens going on holidays.

|Airport|Airport name|Number of trips|
|-------|------------|----------------|
|LAX|Los Angeles International Airport|399995|
|LGA|La Guardia Airport|303773|
|DFW|Dallas Fort Worth International Airport|298524|
|BOS|General Edward Lawrence Logan International Airport|289517|
|ORD|Chicago O'Hare International Airport|286230|
|SFO|San Francisco International Airport|279212|
|CLT|Charlotte Douglas International Airport|270886|
|ATL|Hartsfield Jackson Atlanta International Airport|260891|
|MIA|Miami International Airport|255275|
|PHL|Philadelphia International Airport|234863|

## Most frequent trip

The two most offered routes on Expedia is a trip from Chicago to New York and vice versa.
Only the third most offered trip is a trip to LA, more specifically a trip departing from
New York.

|Route|Number of trips|
|-----|----------------|
|ORD-LGA|19753|
|LGA-ORD|19173|
|JFK-LAX|16314|
|BOS-LGA|15583|
|LGA-BOS|15487|
|JFK-BOS|14659|
|BOS-JFK|14260|
|LAX-JFK|14164|
|SFO-LAX|13853|
|ORD-BOS|13166|
