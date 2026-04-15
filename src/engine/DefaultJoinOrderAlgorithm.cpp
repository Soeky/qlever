// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#include "engine/DefaultJoinOrderAlgorithm.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "util/Exception.h"

namespace qlever::bgp {

JoinPlan DefaultJoinOrderAlgorithm::plan(
    std::span<const SparqlTriple> triples,
    IBgpCardinalityEstimator& estimator, const PlannerContext& ctx) {
  AD_CONTRACT_CHECK(!triples.empty());

  if (triples.size() == 1) {
    return JoinPlan::makeLeaf(0);
  }

  // Estimate single-triple cardinality for each triple.
  EstimatorContext estCtx{ctx.qec_};
  std::vector<std::pair<uint64_t, size_t>> estimates;
  estimates.reserve(triples.size());
  for (size_t i = 0; i < triples.size(); ++i) {
    auto singleTriple = triples.subspan(i, 1);
    uint64_t est = estimator.estimate(singleTriple, estCtx);
    estimates.emplace_back(est, i);
  }

  // Sort by ascending cardinality (stable for determinism).
  std::stable_sort(estimates.begin(), estimates.end());

  // Build a left-deep tree: start with the smallest triple as the
  // leftmost leaf, then join each next-smallest triple on the right.
  JoinPlan current = JoinPlan::makeLeaf(estimates[0].second);
  for (size_t i = 1; i < estimates.size(); ++i) {
    current = JoinPlan::makeJoin(std::move(current),
                                 JoinPlan::makeLeaf(estimates[i].second));
  }

  return current;
}

}  // namespace qlever::bgp
