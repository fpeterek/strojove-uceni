package org.fpeterek.su.clustering.distance

import org.fpeterek.su.clustering.Cluster
import org.fpeterek.su.clustering.distance.DistanceMetric.{Euclidean, Manhattan}

object BestDistance:
  def apply(c1: Cluster, c2: Cluster, metric: DistanceMetric): Double =
    Distances(c1, c2, metric).min
