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

It is also noteworthy that the only airlines that do not appear in multi-segment flights with
other airlines are ultra low cost carriers -- Frontier Airlines (IATA `F9`), Spirit Airlines
(IATA `NK`), and Sun Country Airlines (IATA `SY`). LCCs do not offer multisegment flights
as it incurrs additional costs upon them (i.e. having to take care of passengers in case
of a delay) and they operate their flights at ridiculous times when the airport fees are
the lowest, making it difficult to find a reasonable connecting flight.

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

## Does the date of purchase affect ticket prices

First, we want to compute the difference between the day of the flight and the day the
offer was scraped (a possible date of purchase). Then, we may wish to invert this number
so that the closer to the day of the flight we buy the ticket, the higher this
invented metric is. Airlines generally do not sell tickets more than a year in advance,
thus, we can just subtract the difference in days from 365 to get our desired metric.
We use the inverted metric to get a positive correlation if the price grows the closer
the date of purchase is to the date of the actual flight, and vice versa, making the
results more intuitive.

The first thing we notice is that there is next to no correlation if take into account
all the airlines in our dataset.

```
Correlation for travel class coach: 0.05902518247859265
Correlation for travel class business: -0.09744523681248259
```

However, the results change if only look at the three major US airlines and low cost
carriers separately.

Looking at economy class, the correlation is still slight at best.

```
Correlation for travel class coach and airlines {'AA', 'DL', 'UA'}: 0.0502795446206922
Correlation for travel class coach and airlines {'F9', 'B6', 'NK', 'SY'}: 0.12782560251368322
```

Analyzing business class finally yields interesting results. The closer we get to the date
of the flight, the higher the ticket prices are for traditional major carriers, yet, the
prices of LCCs decrease the closer the date of the flight is.

A possible explanation would be that the main clientele of traditional major carriers are
businesses. Businesses tend to buy tickets close to the date of the flight -- offline meetings
are seldom arranged a year in advance, and since business buy flight tickets to make money,
they do not mind investing in the flight tickets they desperately need. Another possible target
group for traditional carriers providing a luxurious or premium product would be impulsive or
busy rich people.

Low cost carriers are more focused on budget conscious tourists who tend to buy tickets a long
time in advance. Therefore, they are forced to decrease their prices if the aircraft isn't fully
sold out yet to ensure the aircraft takes off completely full and the airline maximizes its
profits.

```
Correlation for travel class business and airlines {'AA', 'DL', 'UA'}: 0.35718535300323284
Correlation for travel class business and airlines {'F9', 'B6', 'NK', 'SY'}: -0.38931697477626526
```

But correlation is but a single real number and thus is incapable of providing us with
an accurate insight into the pricing of flight tickets.

We split the data into buckets by the difference between the date the offer was scraped
and the day of the flight. What buckets were chosen can be seen in the following table.
Ranges are inclusive.

|bucket number|0|1|2|3|4|5|6|7|
|-|-|-|-|-|-|-|-|-|
|days before day of flight|0-1|2-3|4-7|8-14|15-21|22-31|31-61|62-|

First, let's take a look at economy class flights.

![Traditional carriers](./plots/buckets_coach_DL_AA_UA.png)

![Low cost carriers](./plots/buckets_coach_B6_NK_SY_F9.png)

We can see that the, for both types of carriers, the price gradually increases the closer
we get to the day of the flight, however, the increase in price is quite small. The increase
in price is more notable for LCCs, though that can be explained by the fact that LCCs tend
to price their tickets lower and thus there's more room for growth.

A rather curious thing is that the prices do not seem to differ all that much among
LCCs and major traditional carriers, at least in economy.

![Traditional carriers](./plots/buckets_business_DL_AA_UA.png)

![Low cost carriers](./plots/buckets_business_B6_NK_SY_F9.png)

Unlike coach, business class looks a bit more interesting. As we can see from the boxplots,
LCCs keep gradually decreasing the cost of business class tickets as they become more and
more desperate to sell out the entire aircraft.

As was hinted at earlier, traditional major carriers price their business class tickets
reasonably if purchased ahead. The closer we get to the day of the flight, the higher the
cost becomes, to get as much money out of businesses as possible. Only the very last day
before the flight do the prices drop dramatically, to try and convince impulsive buyers
or people looking to upgrade to a higher travel class.

## Does the day of week of the flight affect ticket price

First, let's have a look at economy class.

The plot displays the relation between ticket price and day of week. Days of week start
with Monday (=0) and end with Sunday (=6).

![Traditional carriers](./plots/dow_coach_DL_AA_UA.png)

![Low cost carriers](./plots/dow_coach_B6_NK_SY_F9.png)

As we can see, the prices remain consistent throughout the entire week.

However, things seem to change in business class.

![Traditional carriers](./plots/dow_business_DL_AA_UA.png)

![Low cost carriers](./plots/dow_business_B6_NK_SY_F9.png)

If we look at LCCs, despite the fact that the prices are skewed differently on different
days of the week, the median price remains consistent.

However, that is not the case for traditional carriers. According to the plot, the median ticket
price seems to drop drastically on Tuesday, remains low until Thursday, and only goes back up
on Fridays. Why? A possible answer would be that businesspeople fly out on Mondays and return
on Fridays to get back home for the weekend. Weekends are then reserved for leisurely travel
of affluent people. Fewer people fly during the midweek (as they're either at work or already
on a business trip).

The disparity in pricing between LCCs and traditional carriers is, yet again, caused by
a difference between clientele.



