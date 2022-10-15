package org.fpeterek.su.clustering

import org.fpeterek.su.clustering.Linkage.{Complete, Single}
import org.fpeterek.su.clustering.distance.{BestDistance, DistanceMetric, WorstDistance}

class Clustering(
  val linkage: Linkage,
  val metric: DistanceMetric,
  val numberOfClusters: Int,
):

  implicit class DistanceBetweenClusters(cl: Cluster):
    def distanceFrom(other: Cluster): Double = linkage match
      case Single   => BestDistance(cl, other, metric)
      case Complete => WorstDistance(cl, other, metric)

  private def initClusters(points: Seq[Point]): Seq[Cluster] =
    points.map(p => Cluster(Seq(p)))

  private def findClosest(clusters: Seq[Cluster]): (Int, Int) =
    clusters
      .indices
      .flatMap { i =>
        ((i+1) until clusters.size).map { j =>
          (i, j) -> (clusters(i) distanceFrom clusters(j))
        }
      }
      .maxBy(_._2)
      ._1

  private def joinClusters(clusters: Seq[Cluster], indices: (Int, Int)): Cluster =
    val (i1, i2) = indices
    val fst = clusters(i1)
    val snd = clusters(i2)

    Cluster(fst.points ++ snd.points)

  private def recur(clusters: Seq[Cluster]): Seq[Cluster] =
    val toJoin = findClosest(clusters)
    val joined = joinClusters(clusters, toJoin)

    val newClusters = clusters.indices
      .filter(_ != toJoin._1)
      .filter(_ != toJoin._2)
      .map(clusters) :+ joined

    cluster(newClusters)

  private def cluster(current: Seq[Cluster]): Seq[Cluster] = current.size <= numberOfClusters match
    case true  => current
    case false => recur(current)

  def apply(data: Seq[Point]): Seq[Cluster] = cluster(initClusters(data))
