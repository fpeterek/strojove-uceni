package org.fpeterek.su.clustering.distance

import org.fpeterek.su.clustering.Cluster
import org.fpeterek.su.clustering.distance.DistanceMetric.{Euclidean, Manhattan}


object Distances:
  def apply(c1: Cluster, c2: Cluster, metric: DistanceMetric): Seq[Double] =
    c1.points
      .flatMap { p1 =>
        c2.points.map { p2 =>
          metric match
            case Euclidean => p1 distanceFrom p2
            case Manhattan => p1 manhattanDistance p2
        }
      }
