/**
 * Data models for the DualGraph knowledge graph system.
 * TypeScript port of data_model.py
 */

// ─── Evidence ────────────────────────────────────────────────────────────────

export interface EvidenceNode {
  id: number | null;
  source_title: string;
  source_url: string;
  content: string;
}

export function makeEvidenceNode(
  partial?: Partial<EvidenceNode>,
): EvidenceNode {
  return {
    id: partial?.id ?? null,
    source_title: partial?.source_title ?? "",
    source_url: partial?.source_url ?? "",
    content: partial?.content ?? "",
  };
}

export interface EvidenceNodeList {
  evidence_nodes: EvidenceNode[];
  last_evidence_id: number;
}

export function makeEvidenceNodeList(): EvidenceNodeList {
  return { evidence_nodes: [], last_evidence_id: 0 };
}

export function addEvidenceToList(
  list: EvidenceNodeList,
  source_title: string,
  source_url: string,
  content: string,
): EvidenceNode {
  const node: EvidenceNode = {
    id: list.last_evidence_id + 1,
    source_title,
    source_url,
    content,
  };
  list.evidence_nodes.push(node);
  list.last_evidence_id += 1;
  return node;
}

export function getEvidenceFromList(
  list: EvidenceNodeList,
  id: number,
): EvidenceNode | undefined {
  return list.evidence_nodes.find((n) => n.id === id);
}

// ─── Knowledge Node / Edge ───────────────────────────────────────────────────

export interface KnowledgeNode {
  id: number;
  knowledge: string;
  is_core_entity: boolean;
  cluster_id?: string | null;
  community_id?: string | null;
}

export interface KnowledgeEdge {
  id: number;
  source_id: number;
  target_id: number;
  relation_name: string;
  evidence_nodes: EvidenceNode[];
}

// ─── Chain Types ─────────────────────────────────────────────────────────────

export type ChainType =
  | "enrich"
  | "explore_entity_attribute"
  | "explore_attribute_attribute"
  | "explore_cross_community"
  | "explore_cross_cluster"
  | "explore_sbm_probability"
  | "explore_sbm_entropy"
  | "explore_entity_concept"
  | "explore_entity_concept_coverage_gap"
  | "explore_entity_concept_similarity";

export interface Chain {
  id: number;
  type: ChainType;
  nodes: number[];
  content?: string | null;
  reason: string;
  is_visited: boolean;
}

// ─── Knowledge Graph ─────────────────────────────────────────────────────────

export interface MergedKnowledgeNode {
  original_node_id: number;
  original_name: string;
  merged_node_id: number;
}

export interface SemanticCluster {
  cluster_id: string;
  representative_concept: string;
  source_node_ids: string[];
  similarity_justification: string;
}

export interface CommunityInfo {
  community_id: string;
  representative_node: string;
  source_node_ids: string[];
  size: number;
  detection_method: string;
}

export interface KnowledgeGraph {
  knowledge_nodes: KnowledgeNode[];
  knowledge_edges: KnowledgeEdge[];
  evidence_nodes: EvidenceNode[];
  merged_knowledge_nodes: MergedKnowledgeNode[];
  semantic_clusters: SemanticCluster[];
  community_list: CommunityInfo[];
  embedding_cache: Map<string, number[]>;
  last_evidence_id: number;
  last_knowledge_node_id: number;
  last_knowledge_edge_id: number;
}

export function makeKnowledgeGraph(): KnowledgeGraph {
  return {
    knowledge_nodes: [],
    knowledge_edges: [],
    evidence_nodes: [],
    merged_knowledge_nodes: [],
    semantic_clusters: [],
    community_list: [],
    embedding_cache: new Map(),
    last_evidence_id: 0,
    last_knowledge_node_id: 0,
    last_knowledge_edge_id: 0,
  };
}

// ─── KG helpers ──────────────────────────────────────────────────────────────

export function getKnowledgeNodeById(
  kg: KnowledgeGraph,
  id: number,
): KnowledgeNode | undefined {
  return kg.knowledge_nodes.find((n) => n.id === id);
}

export function getKnowledgeEdgeById(
  kg: KnowledgeGraph,
  id: number,
): KnowledgeEdge | undefined {
  return kg.knowledge_edges.find((e) => e.id === id);
}

export function getEvidenceNodeById(
  kg: KnowledgeGraph,
  id: number,
): EvidenceNode | undefined {
  return kg.evidence_nodes.find((e) => e.id === id);
}

export function addEvidenceNode(
  kg: KnowledgeGraph,
  source_title: string,
  source_url: string,
  content: string,
): EvidenceNode {
  const node: EvidenceNode = {
    id: kg.last_evidence_id + 1,
    source_title,
    source_url,
    content,
  };
  kg.evidence_nodes.push(node);
  kg.last_evidence_id += 1;
  return node;
}

/** Helper: convert string ID to integer (n1 -> 1, e3 -> 3) */
function extractId(idStr: string): number {
  const digits = idStr.replace(/\D/g, "");
  return parseInt(digits, 10);
}

/**
 * Integrate LLM-generated knowledge into the existing knowledge graph.
 */
export function addLlmGeneratedKnowledge(
  kg: KnowledgeGraph,
  llmOutput: {
    new_nodes?: Array<{
      id: string;
      node_name: string;
      is_core_entity: boolean;
    }>;
    new_edges?: Array<{
      id: string;
      source_id: string;
      target_id: string;
      relation_name: string;
    }>;
    evidences_map?: Record<string, string[]>;
  },
  evidenceNodes: EvidenceNode[],
): void {
  const newNodesData = llmOutput.new_nodes ?? [];
  const newEdgesData = llmOutput.new_edges ?? [];
  const evidencesMap = llmOutput.evidences_map ?? {};

  // 1. Process new nodes – deduplicate and add
  const existingNodeNames = new Map<string, number>();
  for (const node of kg.knowledge_nodes) {
    existingNodeNames.set(node.knowledge.toLowerCase(), node.id);
  }
  const nodeIdMapping = new Map<string, number>();
  for (const kn of kg.knowledge_nodes) {
    nodeIdMapping.set(`n${kn.id}`, kn.id);
  }

  for (const nodeData of newNodesData) {
    const rawId = nodeData.id;
    const nodeName = nodeData.node_name.trim();
    const isCore = nodeData.is_core_entity;

    const existingId = existingNodeNames.get(nodeName.toLowerCase());
    if (existingId !== undefined) {
      nodeIdMapping.set(rawId, existingId);
      if (isCore) {
        const node = kg.knowledge_nodes.find((n) => n.id === existingId);
        if (node) node.is_core_entity = true;
      }
    } else {
      const newId = kg.last_knowledge_node_id + 1;
      kg.last_knowledge_node_id = newId;
      nodeIdMapping.set(rawId, newId);
      kg.knowledge_nodes.push({
        id: newId,
        knowledge: nodeName,
        is_core_entity: isCore,
      });
      existingNodeNames.set(nodeName.toLowerCase(), newId);
    }
  }

  // 2. Process new edges
  const existingEdges = new Map<string, number>();
  for (const edge of kg.knowledge_edges) {
    const key = `${edge.source_id}:${edge.target_id}:${edge.relation_name.toLowerCase()}`;
    existingEdges.set(key, edge.id);
  }
  const edgeIdMapping = new Map<string, number>();
  for (const edge of kg.knowledge_edges) {
    edgeIdMapping.set(`e${edge.id}`, edge.id);
  }

  for (const edgeData of newEdgesData) {
    const rawEdgeId = edgeData.id;
    const sourceId = nodeIdMapping.get(edgeData.source_id);
    const targetId = nodeIdMapping.get(edgeData.target_id);
    if (sourceId === undefined || targetId === undefined) continue;

    const relation = edgeData.relation_name.trim();
    const edgeKey = `${sourceId}:${targetId}:${relation.toLowerCase()}`;
    const existingEdgeId = existingEdges.get(edgeKey);

    if (existingEdgeId !== undefined) {
      edgeIdMapping.set(rawEdgeId, existingEdgeId);
    } else {
      const newEdgeId = kg.last_knowledge_edge_id + 1;
      kg.last_knowledge_edge_id = newEdgeId;
      edgeIdMapping.set(rawEdgeId, newEdgeId);
      kg.knowledge_edges.push({
        id: newEdgeId,
        source_id: sourceId,
        target_id: targetId,
        relation_name: relation,
        evidence_nodes: [],
      });
      existingEdges.set(edgeKey, newEdgeId);
    }
  }

  // 3. Process evidence mapping
  const enToEvidence = new Map<string, EvidenceNode>();
  for (const enId of Object.keys(evidencesMap)) {
    if (enId.startsWith("EN")) {
      const num = parseInt(enId.slice(2), 10);
      if (!isNaN(num)) {
        const matched = evidenceNodes.find((e) => e.id === num);
        if (matched) enToEvidence.set(enId, matched);
      }
    }
  }

  // 4. Add evidence support to each edge
  for (const [enId, edgeIds] of Object.entries(evidencesMap)) {
    const evidenceNode = enToEvidence.get(enId);
    if (!evidenceNode || evidenceNode.id === null) continue;

    for (const rawEdgeId of edgeIds) {
      const edgeId = edgeIdMapping.get(rawEdgeId);
      if (edgeId === undefined) continue;

      const edge = kg.knowledge_edges.find((e) => e.id === edgeId);
      if (edge) {
        if (!edge.evidence_nodes.some((e) => e.id === evidenceNode.id)) {
          edge.evidence_nodes.push(evidenceNode);
        }
      }
    }
  }

  // 5. Clean up orphaned concept nodes
  removeOrphanedConceptNodes(kg);
}

function removeOrphanedConceptNodes(kg: KnowledgeGraph): void {
  const activeNodeIds = new Set<number>();
  for (const edge of kg.knowledge_edges) {
    activeNodeIds.add(edge.source_id);
    activeNodeIds.add(edge.target_id);
  }
  kg.knowledge_nodes = kg.knowledge_nodes.filter(
    (node) => node.is_core_entity || activeNodeIds.has(node.id),
  );
}

// ─── KG text/JSON serialization ──────────────────────────────────────────────

export function knowledgeGraphToText(kg: KnowledgeGraph): string {
  const lines: string[] = [];

  lines.push("[KNOWLEDGE NODES]");
  const sortedNodes = [...kg.knowledge_nodes].sort((a, b) => a.id - b.id);
  for (const node of sortedNodes) {
    const typeLabel = node.is_core_entity ? "entity" : "attribute";
    lines.push(`KN${node.id}: ${node.knowledge} [type: ${typeLabel}]`);
  }
  lines.push("");

  lines.push("[RELATIONSHIPS]");
  const sortedEdges = [...kg.knowledge_edges].sort((a, b) => {
    if (a.source_id !== b.source_id) return a.source_id - b.source_id;
    if (a.target_id !== b.target_id) return a.target_id - b.target_id;
    return a.relation_name.localeCompare(b.relation_name);
  });
  for (const edge of sortedEdges) {
    lines.push(
      `KN${edge.source_id} → "${edge.relation_name}" → KN${edge.target_id}`,
    );
  }
  lines.push("");

  if (kg.semantic_clusters.length > 0) {
    lines.push("[SEMANTIC CLUSTERS]");
    for (let i = 0; i < kg.semantic_clusters.length; i++) {
      const cluster = kg.semantic_clusters[i];
      lines.push(
        `Cluster ${cluster.cluster_id}: ${cluster.representative_concept}`,
      );
      lines.push(`  Nodes: ${cluster.source_node_ids.join(", ")}`);
      if (cluster.similarity_justification) {
        lines.push(
          `  Justification: ${cluster.similarity_justification}`,
        );
      }
    }
    lines.push("");
  }

  if (kg.community_list.length > 0) {
    lines.push("[COMMUNITIES]");
    for (const community of kg.community_list) {
      lines.push(
        `Community ${community.community_id}: ${community.representative_node} [method: ${community.detection_method}, size: ${community.size}]`,
      );
      lines.push(`  Nodes: ${community.source_node_ids.join(", ")}`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

export function knowledgeGraphToJson(
  kg: KnowledgeGraph,
): Record<string, unknown> {
  const id2node = new Map<number, KnowledgeNode>();
  for (const node of kg.knowledge_nodes) {
    id2node.set(node.id, node);
  }
  return {
    knowledge_nodes: kg.knowledge_nodes.map((node) => ({
      node_id: `n${node.id}`,
      knowledge: node.knowledge,
      is_core_entity: node.is_core_entity,
    })),
    knowledge_edges: kg.knowledge_edges.map((edge) => ({
      edge_id: `e${edge.id}`,
      representation: `${id2node.get(edge.source_id)?.knowledge ?? "?"} - ${edge.relation_name} -> ${id2node.get(edge.target_id)?.knowledge ?? "?"}`,
    })),
  };
}

export function knowledgeGraphToTextForWriter(
  kg: KnowledgeGraph,
  evidenceNodeIds: Set<number>,
): string {
  const id2node = new Map<number, KnowledgeNode>();
  for (const node of kg.knowledge_nodes) id2node.set(node.id, node);

  const lines: string[] = [];
  lines.push("[Core Entity Nodes]");
  for (const node of kg.knowledge_nodes) {
    if (node.is_core_entity) lines.push(node.knowledge);
  }
  lines.push("");

  lines.push("[Relationships]");
  for (const edge of kg.knowledge_edges) {
    const edgeEvidenceIds = edge.evidence_nodes
      .map((en) => en.id)
      .filter((id) => id !== null) as number[];
    if (edgeEvidenceIds.some((eid) => evidenceNodeIds.has(eid))) {
      const src = id2node.get(edge.source_id)?.knowledge ?? "?";
      const tgt = id2node.get(edge.target_id)?.knowledge ?? "?";
      lines.push(`${src} - ${edge.relation_name} -> ${tgt}`);
    }
  }
  return lines.join("\n");
}

export function getClusterMap(kg: KnowledgeGraph): Map<number, number> {
  const map = new Map<number, number>();
  for (const m of kg.merged_knowledge_nodes) {
    map.set(m.original_node_id, m.merged_node_id);
  }
  return map;
}

/**
 * Apply merge/clustering results to the knowledge graph.
 */
export function applyMergeNodeResults(
  kg: KnowledgeGraph,
  clusteringResults: {
    clusters?: Array<{
      cluster_id: string;
      representative_concept: string;
      source_node_ids: string[];
      similarity_justification: string;
    }>;
  },
): void {
  if (
    !clusteringResults ||
    !clusteringResults.clusters ||
    clusteringResults.clusters.length === 0
  )
    return;

  const nodeReplacementMap = new Map<number, number>();
  const nodesToRemove = new Set<number>();

  for (const cluster of clusteringResults.clusters) {
    const sourceNodeIds: number[] = [];
    for (const nidStr of cluster.source_node_ids) {
      sourceNodeIds.push(
        nidStr.startsWith("n") ? parseInt(nidStr.slice(1), 10) : parseInt(nidStr, 10),
      );
    }
    if (sourceNodeIds.length < 2) continue;

    const validSourceNodes: KnowledgeNode[] = [];
    for (const nodeId of sourceNodeIds) {
      const node = getKnowledgeNodeById(kg, nodeId);
      if (node && !node.is_core_entity) {
        validSourceNodes.push(node);
        nodesToRemove.add(nodeId);
      }
    }
    if (validSourceNodes.length < 2) continue;

    const newNodeId = kg.last_knowledge_node_id + 1;
    kg.last_knowledge_node_id = newNodeId;

    kg.knowledge_nodes.push({
      id: newNodeId,
      knowledge: cluster.representative_concept,
      is_core_entity: false,
    });

    for (const node of validSourceNodes) {
      kg.merged_knowledge_nodes.push({
        original_node_id: node.id,
        original_name: node.knowledge,
        merged_node_id: newNodeId,
      });
    }

    for (const nodeId of sourceNodeIds) {
      nodeReplacementMap.set(nodeId, newNodeId);
    }
  }

  if (nodeReplacementMap.size === 0) return;

  // Process edges
  const newEdgeMap = new Map<string, KnowledgeEdge>();
  const edgesToRemove = new Set<number>();

  for (const edge of kg.knowledge_edges) {
    const newSourceId =
      nodeReplacementMap.get(edge.source_id) ?? edge.source_id;
    const newTargetId =
      nodeReplacementMap.get(edge.target_id) ?? edge.target_id;

    if (newSourceId === newTargetId) {
      edgesToRemove.add(edge.id);
      continue;
    }

    if (
      nodeReplacementMap.has(edge.source_id) ||
      nodeReplacementMap.has(edge.target_id)
    ) {
      edgesToRemove.add(edge.id);
      const edgeKey = `${newSourceId}:${newTargetId}`;

      const existing = newEdgeMap.get(edgeKey);
      if (existing) {
        for (const ev of edge.evidence_nodes) {
          if (!existing.evidence_nodes.some((e) => e.id === ev.id)) {
            existing.evidence_nodes.push(ev);
          }
        }
      } else {
        const newEdgeId = kg.last_knowledge_edge_id + 1;
        kg.last_knowledge_edge_id = newEdgeId;
        newEdgeMap.set(edgeKey, {
          id: newEdgeId,
          source_id: newSourceId,
          target_id: newTargetId,
          relation_name: edge.relation_name,
          evidence_nodes: [...edge.evidence_nodes],
        });
      }
    }
  }

  // Update edges list
  kg.knowledge_edges = kg.knowledge_edges.filter(
    (e) => !edgesToRemove.has(e.id),
  );
  kg.knowledge_edges.push(...newEdgeMap.values());

  // Remove source nodes
  kg.knowledge_nodes = kg.knowledge_nodes.filter(
    (n) => !nodesToRemove.has(n.id),
  );

  // Clean up orphaned edges
  const validNodeIds = new Set(kg.knowledge_nodes.map((n) => n.id));
  kg.knowledge_edges = kg.knowledge_edges.filter(
    (e) => validNodeIds.has(e.source_id) && validNodeIds.has(e.target_id),
  );
}

/**
 * Apply semantic clustering results (non-destructive).
 */
export function applySemanticClusteringResults(
  kg: KnowledgeGraph,
  clusteringResults: {
    clusters?: Array<{
      cluster_id: string;
      representative_concept: string;
      source_node_ids: string[];
      similarity_justification: string;
    }>;
  },
): void {
  if (!clusteringResults?.clusters?.length) return;

  const normalizedClusters: SemanticCluster[] = [];
  const nodeToCluster = new Map<number, string>();

  for (const cluster of clusteringResults.clusters) {
    const clusterId =
      (cluster.cluster_id ?? "").trim() || "semantic_cluster";
    const rep = (cluster.representative_concept ?? "").trim();
    const justification =
      (cluster.similarity_justification ?? "").trim();

    const srcNodeIds: string[] = [];
    const srcNodeInts: number[] = [];
    for (let s of cluster.source_node_ids ?? []) {
      s = s.trim();
      if (!s) continue;
      srcNodeIds.push(s.startsWith("n") ? s : `n${s}`);
      const digits = s.replace(/\D/g, "");
      if (digits) srcNodeInts.push(parseInt(digits, 10));
    }

    if (srcNodeInts.length < 2) continue;

    normalizedClusters.push({
      cluster_id: clusterId,
      representative_concept: rep,
      source_node_ids: srcNodeIds,
      similarity_justification: justification,
    });

    for (const nid of srcNodeInts) {
      nodeToCluster.set(nid, clusterId);
    }
  }

  for (const node of kg.knowledge_nodes) {
    if (node.is_core_entity) continue;
    const cid = nodeToCluster.get(node.id);
    if (cid !== undefined) node.cluster_id = cid;
  }

  kg.semantic_clusters = normalizedClusters;
}

/**
 * Apply community detection results (non-destructive).
 */
export function applyCommunityDetectionResults(
  kg: KnowledgeGraph,
  communityResults: Map<number, string>,
): void {
  if (!communityResults || communityResults.size === 0) return;

  const communityToNodes = new Map<string, number[]>();
  for (const [nodeId, communityId] of communityResults) {
    const list = communityToNodes.get(communityId) ?? [];
    list.push(nodeId);
    communityToNodes.set(communityId, list);
  }

  const normalizedCommunities: CommunityInfo[] = [];
  const sortedEntries = [...communityToNodes.entries()].sort(
    (a, b) => a[0].localeCompare(b[0]),
  );

  for (const [communityId, nodeIds] of sortedEntries) {
    if (nodeIds.length < 1) continue;

    const nodeKnowledges: string[] = [];
    for (const node of kg.knowledge_nodes) {
      if (nodeIds.includes(node.id)) {
        nodeKnowledges.push(node.knowledge);
      }
    }
    const representative = nodeKnowledges[0] ?? "";

    normalizedCommunities.push({
      community_id: communityId,
      representative_node: representative,
      source_node_ids: nodeIds.sort((a, b) => a - b).map((id) => `n${id}`),
      size: nodeIds.length,
      detection_method: "Leiden",
    });

    for (const node of kg.knowledge_nodes) {
      if (nodeIds.includes(node.id)) {
        node.community_id = communityId;
      }
    }
  }

  kg.community_list = normalizedCommunities;
}
