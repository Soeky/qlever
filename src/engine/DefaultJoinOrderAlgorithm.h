// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_ENGINE_DEFAULTJOINORDERALGORITHM_H
#define QLEVER_SRC_ENGINE_DEFAULTJOINORDERALGORITHM_H

#include "engine/IJoinOrderAlgorithm.h"

namespace qlever::bgp {

// Simple left-deep join ordering by ascending single-triple cardinality.
// Demonstrates the IJoinOrderAlgorithm interface. The actual "default"
// path in QueryPlanner::optimizeCommutatively() continues to call
// fillDpTab() directly — this class is for the plugin path only.
class DefaultJoinOrderAlgorithm : public IJoinOrderAlgorithm {
 public:
  JoinPlan plan(std::span<const SparqlTriple> triples,
                IBgpCardinalityEstimator& estimator,
                const PlannerContext& ctx) override;
};

}  // namespace qlever::bgp

#endif  // QLEVER_SRC_ENGINE_DEFAULTJOINORDERALGORITHM_H
