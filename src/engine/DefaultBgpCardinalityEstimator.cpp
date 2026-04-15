// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#include "engine/DefaultBgpCardinalityEstimator.h"

#include "engine/IndexScan.h"
#include "engine/QueryExecutionContext.h"
#include "parser/SparqlTriple.h"
#include "util/Exception.h"

namespace qlever::bgp {

uint64_t DefaultBgpCardinalityEstimator::estimate(
    std::span<const SparqlTriple> triples, const EstimatorContext& ctx) {
  AD_CONTRACT_CHECK(ctx.qec_ != nullptr);

  if (triples.size() == 1) {
    // For a single triple, create a temporary IndexScan to get the
    // native QLever size estimate.
    const auto& triple = triples[0];
    auto simpleTriple = triple.getSimple();

    // Count variables to choose a base permutation, matching the
    // heuristic from seedFromOrdinaryTriple.
    const bool sVar = simpleTriple.s_.isVariable();
    const bool pVar = simpleTriple.p_.isVariable();
    const bool oVar = simpleTriple.o_.isVariable();
    const size_t numVars =
        static_cast<size_t>(sVar) + static_cast<size_t>(pVar) +
        static_cast<size_t>(oVar);

    using enum Permutation::Enum;
    Permutation::Enum perm = PSO;
    if (numVars == 1) {
      if (sVar) {
        perm = POS;
      } else if (oVar) {
        perm = PSO;
      } else {
        perm = SPO;
      }
    } else if (numVars == 2) {
      if (!pVar) {
        perm = PSO;
      } else if (!sVar) {
        perm = SPO;
      } else {
        perm = OSP;
      }
    }
    // numVars == 0 or 3: PSO is fine.

    IndexScan scan(ctx.qec_, perm, simpleTriple);
    return scan.getSizeEstimate();
  }

  // Multi-triple estimation is not implemented in Phase 2.
  // Phase 4 will provide MinCardinalityBgpEstimator for this case.
  AD_THROW(
      "DefaultBgpCardinalityEstimator does not support multi-triple "
      "estimation. Use a custom estimator for join cardinality.");
}

}  // namespace qlever::bgp
