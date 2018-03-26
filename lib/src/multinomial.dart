part of grizzly.naive_bayes;

/// Multinomial naive bayes estimator
class MultinomialNB implements DiscreteNaiveBayes {
  const MultinomialNB();

  MultinomialNBModel<LT> fit<LT>(Numeric2DView<double> x, ArrayView y,
      {ArrayView labels,
      NumericSeries<LT, double> labelPriori,
      bool fitPriori: false,
      double alpha: 1.0}) {
    checkXY(x, y);
    final int numFeatures = x.numCols;
    labels ??= y.unique();
    Double2DFix yBinarized = labelBinarize(y, labels: labels).toDouble;

    // TODO what if there is only one label?

    // TODO sample weights

    Double1DFix labelCounts = new Double1DFix.sized(labels.length);
    Double2DFix featureCounts =
        new Double2DFix.sized(labels.length, numFeatures);

    count(x, yBinarized, labelCounts, featureCounts);
    // TODO checkAlpha?
    Double2D featureLB = featureLogProbability(featureCounts, alpha: alpha);

    NumericSeries<LT, double> labelLogPriori;
    if (labelPriori != null) {
      if (!labelPriori.labelsMatch(labels)) throw new Exception('Priori ');
      labelLogPriori = labelPriori.log;
    } else if (fitPriori) {
      // TODO
    } else {
      // TODO
    }

    return new MultinomialNBModel<LT>(featureLB, labelLogPriori);
  }

  /// Counts and smooths feature occurrences
  static void count(Numeric2DView<double> x, Double2DView yBinarized,
      Double1DFix labelCounts, Double2DFix featureCounts) {
    labelCounts.assign(yBinarized.col.sum);
    featureCounts.assign(yBinarized.transpose * x);
  }

  /// Apply smoothing to raw counts and recompute log probabilities
  static Double2D featureLogProbability(Double2DFix featureCounts,
      {double alpha: 1.0}) {
    Double2D smoothedFC = featureCounts + alpha;
    Double1D smoothedCC = smoothedFC.row.sum;
    return smoothedFC.log.col - smoothedCC.log;
  }
}

/// Multinomial naive bayes model
class MultinomialNBModel<LT> implements NaiveBayesModel<LT> {
  final Double2D featureLogProbability;

  final NumericSeries<LT, double> labelLogPriori;

  MultinomialNBModel(this.featureLogProbability, this.labelLogPriori);

  /// Perform classification on an array of test vectors [x].
  Array<LT> predict(Numeric2DView<double> x) {
    Double2D jll = x * featureLogProbability.transpose + labelLogPriori.data;
    return labelLogPriori.labels.pickByIndices(jll.row.argMax);
  }
}
