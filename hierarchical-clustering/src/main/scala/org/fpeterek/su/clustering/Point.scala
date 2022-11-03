package org.fpeterek.su.clustering

case class Point(
  x: Double,
  y: Double,
):
  def distanceFrom(other: Point): Double =
    math.sqrt(math.pow(x - other.x, 2) + math.pow(y - other.y, 2))

  def manhattanDistance(other: Point): Double =
    math.abs(x - other.x) + math.abs(y - other.y)
