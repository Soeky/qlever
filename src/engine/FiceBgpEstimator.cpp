// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifdef QLEVER_WITH_FICE

#include "engine/FiceBgpEstimator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "parser/SparqlTriple.h"
#include "util/Exception.h"
#include "util/Log.h"

namespace qlever::bgp {

// ---------------------------------------------------------------------------
// MmapFile implementation
// ---------------------------------------------------------------------------

MmapFile::~MmapFile() {
  if (data && data != MAP_FAILED) munmap(data, size);
  if (fd >= 0) close(fd);
}

MmapFile::MmapFile(MmapFile&& o) noexcept
    : data(o.data), size(o.size), fd(o.fd) {
  o.data = nullptr;
  o.size = 0;
  o.fd = -1;
}

MmapFile& MmapFile::operator=(MmapFile&& o) noexcept {
  if (this != &o) {
    if (data && data != MAP_FAILED) munmap(data, size);
    if (fd >= 0) close(fd);
    data = o.data;
    size = o.size;
    fd = o.fd;
    o.data = nullptr;
    o.size = 0;
    o.fd = -1;
  }
  return *this;
}

MmapFile mmapOpen(const std::string& path) {
  MmapFile m;
  m.fd = open(path.c_str(), O_RDONLY);
  if (m.fd < 0) {
    throw std::runtime_error("FiceBgpEstimator: cannot open " + path);
  }
  struct stat st;
  fstat(m.fd, &st);
  m.size = st.st_size;
  m.data = mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, m.fd, 0);
  if (m.data == MAP_FAILED) {
    close(m.fd);
    m.fd = -1;
    throw std::runtime_error("FiceBgpEstimator: mmap failed for " + path);
  }
  return m;
}

// ---------------------------------------------------------------------------
// Minimal JSON parser — just enough for metadata.json and uri_to_id.json
// ---------------------------------------------------------------------------

namespace {

struct JsonValue {
  using ObjMap = std::unordered_map<std::string, JsonValue>;
  enum Type { STRING, NUMBER, BOOL, OBJECT, NONE } type = NONE;
  std::string strVal;
  double numVal = 0;
  std::unique_ptr<ObjMap> objVal;

  int asInt() const { return static_cast<int>(numVal); }
  const std::string& asString() const { return strVal; }
  const JsonValue& operator[](const std::string& key) const {
    return objVal->at(key);
  }
};

void skipWs(const std::string& s, size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' ||
                           s[i] == '\t'))
    i++;
}

std::string parseJsonString(const std::string& s, size_t& i) {
  i++;  // skip "
  std::string result;
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\') {
      i++;
      if (i < s.size()) result += s[i];
    } else {
      result += s[i];
    }
    i++;
  }
  i++;  // skip "
  return result;
}

JsonValue parseJson(const std::string& s, size_t& i) {
  skipWs(s, i);
  JsonValue v;
  if (i >= s.size()) return v;
  if (s[i] == '"') {
    v.type = JsonValue::STRING;
    v.strVal = parseJsonString(s, i);
  } else if (s[i] == '{') {
    v.type = JsonValue::OBJECT;
    v.objVal = std::make_unique<JsonValue::ObjMap>();
    i++;
    skipWs(s, i);
    while (i < s.size() && s[i] != '}') {
      skipWs(s, i);
      std::string key = parseJsonString(s, i);
      skipWs(s, i);
      i++;  // skip :
      (*v.objVal)[key] = parseJson(s, i);
      skipWs(s, i);
      if (i < s.size() && s[i] == ',') i++;
    }
    if (i < s.size()) i++;  // skip }
  } else if (s[i] == 't' || s[i] == 'f') {
    v.type = JsonValue::BOOL;
    if (s.substr(i, 4) == "true") {
      i += 4;
    } else {
      i += 5;
    }
  } else {
    v.type = JsonValue::NUMBER;
    size_t start = i;
    while (i < s.size() && (std::isdigit(s[i]) || s[i] == '.' ||
                             s[i] == '-' || s[i] == 'e' || s[i] == 'E' ||
                             s[i] == '+'))
      i++;
    v.numVal = std::stod(s.substr(start, i - start));
  }
  return v;
}

JsonValue loadJsonFile(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("FiceBgpEstimator: cannot open " + path);
  }
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  size_t i = 0;
  return parseJson(content, i);
}

}  // namespace

// ---------------------------------------------------------------------------
// FiceBgpEstimator
// ---------------------------------------------------------------------------

FiceBgpEstimator::FiceBgpEstimator(const std::string& artifactsDir) {
  // 1. Load metadata
  auto metadata = loadJsonFile(artifactsDir + "/metadata.json");
  embeddingDim_ = metadata["embedding_dim"].asInt();
  featDim_ = metadata["feat_dim"].asInt();
  graphNumEdges_ = metadata["graph_num_edges"].asInt();

  LOG(INFO) << "FICE: graph has " << metadata["num_entities"].asInt()
            << " entities, " << metadata["num_relations"].asInt()
            << " relations, " << graphNumEdges_ << " edges" << std::endl;

  // 2. Load TorchScript model (prefer scripted over traced)
  std::string modelPath = artifactsDir + "/query_gnn_scripted.pt";
  {
    std::ifstream test(modelPath);
    if (!test.good()) {
      modelPath = artifactsDir + "/query_gnn_traced.pt";
    }
  }
  LOG(INFO) << "FICE: loading model from " << modelPath << std::endl;
  model_ = torch::jit::load(modelPath);
  model_.eval();

  // 3. Load URI-to-ID mappings
  auto uriJson = loadJsonFile(artifactsDir + "/uri_to_id.json");
  for (auto& [uri, val] : *uriJson["entities"].objVal) {
    entityUriToId_[uri] = val.asInt();
  }
  for (auto& [uri, val] : *uriJson["relations"].objVal) {
    relationUriToId_[uri] = val.asInt();
  }
  LOG(INFO) << "FICE: loaded " << entityUriToId_.size() << " entity URIs, "
            << relationUriToId_.size() << " relation URIs" << std::endl;

  // 4. Memory-map embeddings and occurrence counts
  entityEmbMmap_ = mmapOpen(artifactsDir + "/entity_embeddings.bin");
  relationEmbMmap_ = mmapOpen(artifactsDir + "/relation_embeddings.bin");
  entityOccMmap_ = mmapOpen(artifactsDir + "/entity_occurrences.bin");
  relationOccMmap_ = mmapOpen(artifactsDir + "/relation_occurrences.bin");

  entityEmb_ = static_cast<const float*>(entityEmbMmap_.data);
  relationEmb_ = static_cast<const float*>(relationEmbMmap_.data);
  entityOcc_ = static_cast<const float*>(entityOccMmap_.data);
  relationOcc_ = static_cast<const float*>(relationOccMmap_.data);

  LOG(INFO) << "FICE: estimator ready" << std::endl;
}

std::string FiceBgpEstimator::extractUri(const TripleComponent& tc) {
  if (tc.isIri()) {
    // toStringRepresentation() returns "<http://...>", strip angle brackets.
    std::string repr = tc.getIri().toStringRepresentation();
    if (repr.size() >= 2 && repr.front() == '<' && repr.back() == '>') {
      return repr.substr(1, repr.size() - 2);
    }
    return repr;
  }
  if (tc.isVariable()) {
    return "?" + tc.getVariable().name();
  }
  // Literal or other: treat as unknown entity
  return "";
}

uint64_t FiceBgpEstimator::estimate(std::span<const SparqlTriple> triples,
                                     const EstimatorContext& /*ctx*/) {
  if (triples.empty()) return 0;

  // Assign local node IDs: collect unique subjects and objects.
  std::unordered_map<std::string, int> nodeMap;
  auto getNodeId = [&](const std::string& name) -> int {
    auto it = nodeMap.find(name);
    if (it != nodeMap.end()) return it->second;
    int id = static_cast<int>(nodeMap.size());
    nodeMap[name] = id;
    return id;
  };

  struct ParsedTriple {
    int srcNode;
    int dstNode;
    int relationId;  // -1 if not found
    std::string predUri;
  };
  std::vector<ParsedTriple> parsed;
  parsed.reserve(triples.size());

  for (const auto& triple : triples) {
    std::string sUri = extractUri(triple.s_);
    std::string oUri = extractUri(triple.o_);

    // Extract predicate URI from the PropertyPath.
    std::string pUri;
    auto simplePred = triple.getSimplePredicate();
    if (simplePred.has_value()) {
      std::string repr(simplePred.value());
      if (repr.size() >= 2 && repr.front() == '<' && repr.back() == '>') {
        pUri = repr.substr(1, repr.size() - 2);
      } else {
        pUri = repr;
      }
    } else if (std::holds_alternative<Variable>(triple.p_)) {
      pUri = "?" + std::get<Variable>(triple.p_).name();
    }

    int s = getNodeId(sUri);
    int o = getNodeId(oUri);

    auto rit = relationUriToId_.find(pUri);
    int rid = (rit != relationUriToId_.end()) ? rit->second : -1;

    parsed.push_back({s, o, rid, pUri});
  }

  int numNodes = static_cast<int>(nodeMap.size());
  int numEdges = static_cast<int>(parsed.size());

  // Build node features: [occ(3) | embedding(D)]
  auto x = torch::zeros({numNodes, featDim_});
  auto xAcc = x.accessor<float, 2>();

  for (auto& [name, localId] : nodeMap) {
    bool isVar = (!name.empty() && name[0] == '?');
    if (!isVar) {
      auto eit = entityUriToId_.find(name);
      if (eit != entityUriToId_.end()) {
        int eid = eit->second;
        xAcc[localId][0] = entityOcc_[eid * 3 + 0];
        xAcc[localId][1] = entityOcc_[eid * 3 + 1];
        xAcc[localId][2] = entityOcc_[eid * 3 + 2];
        for (int d = 0; d < embeddingDim_; d++) {
          xAcc[localId][3 + d] = entityEmb_[eid * embeddingDim_ + d];
        }
      }
      // Unknown entity: zero embedding + zero counts (already initialized)
    }
    // Variables: zero embedding + zero counts (already initialized)
  }

  // Build edge features: [occ(3) | embedding(D)]
  auto edgeAttr = torch::zeros({numEdges, featDim_});
  auto eaAcc = edgeAttr.accessor<float, 2>();

  for (int e = 0; e < numEdges; e++) {
    int rid = parsed[e].relationId;
    if (rid >= 0) {
      eaAcc[e][0] = relationOcc_[rid * 3 + 0];
      eaAcc[e][1] = relationOcc_[rid * 3 + 1];
      eaAcc[e][2] = relationOcc_[rid * 3 + 2];
      for (int d = 0; d < embeddingDim_; d++) {
        eaAcc[e][3 + d] = relationEmb_[rid * embeddingDim_ + d];
      }
    }
  }

  // Build edge_index
  auto edgeIndex = torch::zeros({2, numEdges}, torch::kLong);
  auto eiAcc = edgeIndex.accessor<int64_t, 2>();
  for (int e = 0; e < numEdges; e++) {
    eiAcc[0][e] = parsed[e].srcNode;
    eiAcc[1][e] = parsed[e].dstNode;
  }

  // Batch tensor: single query → all zeros
  auto batch = torch::zeros({numNodes}, torch::kLong);

  // Graph size
  auto graphSize =
      torch::tensor({static_cast<float>(graphNumEdges_)});

  // Apply log1p to occurrence counts (matching training behavior)
  x.slice(1, 0, 3) = torch::log1p(x.slice(1, 0, 3));
  edgeAttr.slice(1, 0, 3) = torch::log1p(edgeAttr.slice(1, 0, 3));

  // Run forward pass
  torch::NoGradGuard noGrad;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(x);
  inputs.push_back(edgeIndex);
  inputs.push_back(edgeAttr);
  inputs.push_back(batch);
  inputs.push_back(graphSize);

  auto output = model_.forward(inputs).toTensor();
  float logCardinality = output.item<float>();

  // Convert from log-space: expm1(x) = exp(x) - 1
  float cardinality = std::expm1(logCardinality);
  if (cardinality < 1.0f) cardinality = 1.0f;

  return static_cast<uint64_t>(cardinality);
}

void FiceBgpEstimator::warmup(const EstimatorContext& /*ctx*/) {
  // Run a dummy inference to warm up the JIT compiler.
  auto x = torch::zeros({2, featDim_});
  auto edgeIndex = torch::tensor({{0L}, {1L}});
  auto edgeAttr = torch::zeros({1, featDim_});
  auto batch = torch::zeros({2}, torch::kLong);
  auto graphSize = torch::tensor({static_cast<float>(graphNumEdges_)});

  torch::NoGradGuard noGrad;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(x);
  inputs.push_back(edgeIndex);
  inputs.push_back(edgeAttr);
  inputs.push_back(batch);
  inputs.push_back(graphSize);

  model_.forward(inputs);
  LOG(INFO) << "FICE: warmup complete" << std::endl;
}

}  // namespace qlever::bgp

#endif  // QLEVER_WITH_FICE
