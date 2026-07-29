/**
 * Recursively flattens a saved model group into an ordered list of raw model
 * names, expanding any nested group members along the way.
 *
 * The server rejects cycles on save, but `seen` still guards against one
 * slipping through (e.g. a group deleted/recreated between fetches) so this
 * can't recurse forever — it just silently stops expanding that branch.
 *
 * @param {string} groupUuid
 * @param {Map<string, {members: Array<{order: number, modelName: ?string, childGroupUuid: ?string}>}>} groupsMap
 * @param {Set<string>} [seen]
 * @returns {string[]}
 */
export function expandModelGroupMembers(groupUuid, groupsMap, seen = new Set()) {
  if (seen.has(groupUuid)) return [];
  seen.add(groupUuid);

  const group = groupsMap.get(groupUuid);
  if (!group) return [];

  const members = [...group.members].sort((a, b) => a.order - b.order);
  const out = [];
  for (const member of members) {
    if (member.modelName) {
      out.push(member.modelName);
    } else if (member.childGroupUuid) {
      out.push(...expandModelGroupMembers(member.childGroupUuid, groupsMap, seen));
    }
  }
  return out;
}
